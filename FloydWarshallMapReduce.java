import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.FileUtil;

import java.io.IOException;

public class FloydWarshallMapReduce {

    public static class FloydWarshallMapper extends Mapper<Object, Text, Text, Text> {
        @Override
        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            String[] parts = value.toString().split(",");

            // Ensure the line has exactly three parts
            if (parts.length < 3) {
                System.err.println("Invalid line: " + value.toString());
                return; // Skip this line
            }

            String u = parts[0];
            String v = parts[1];
            String distance = parts[2];

            // Direct path emission
            context.write(new Text(u + "," + v), new Text(distance));

            // Emit possible paths through the current intermediate vertex 'k'
            String intermediateVertex = context.getConfiguration().get("intermediateVertex");
            if (!u.equals(intermediateVertex) && !v.equals(intermediateVertex)) {
                // Emit possible path through intermediate vertex 'k'
                context.write(new Text(u + "," + v), new Text(u + "," + intermediateVertex + "," + v + "," + distance));
            }
        }
    }

public static class FloydWarshallReducer extends Reducer<Text, Text, Text, Text> {
    private boolean isLastIteration = false;

    @Override
    protected void setup(Context context) throws IOException, InterruptedException {
        Configuration conf = context.getConfiguration();
        String vertices = conf.get("vertices");
        String intermediateVertex = conf.get("intermediateVertex");
        String[] vertexArray = vertices.split(",");
        isLastIteration = intermediateVertex.equals(vertexArray[vertexArray.length - 1]);
    }

    @Override
    public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
        int minDistance = Integer.MAX_VALUE;

        for (Text value : values) {
            String[] parts = value.toString().split(",");
            if (parts.length == 1) {
                // Direct path distance
                minDistance = Math.min(minDistance, Integer.parseInt(parts[0]));
            } else {
                // Path with intermediate vertex
                int distanceThroughK = Integer.parseInt(parts[3]);
                minDistance = Math.min(minDistance, distanceThroughK);
            }
        }

        // Generate output in the requested format
        String[] uv = key.toString().split(",");
        String u = uv[0];
        String v = uv[1];

        if (isLastIteration) {
            // Format for the last iteration
            context.write(new Text("Shortest path from " + u + " to " + v), new Text("Distance: " + minDistance));
        } else {
            // Format for intermediate iterations
            context.write(new Text(u + "," + v), new Text(String.valueOf(minDistance)));
        }
    }
}



    public static void main(String[] args) throws Exception {
        if (args.length < 3) {
            System.err.println("Usage: FloydWarshallMapReduce <input path> <output path> <vertices>");
            System.exit(-1);
        }

        String vertices = args[2]; // List of all vertices (e.g., "A,B,C,D,E,F")
        Configuration conf = new Configuration();
        conf.set("vertices", vertices);
        conf.set("mapreduce.output.textoutputformat.separator", ",");

        // Start with the initial input path
        String currentInputPath = args[0];

        // Iterate over each intermediate vertex
        String[] vertexArray = vertices.split(",");
        for (String k : vertexArray) {
            conf.set("intermediateVertex", k);

            Job job = Job.getInstance(conf, "FloydWarshall - Intermediate Vertex: " + k);
            job.setJarByClass(FloydWarshallMapReduce.class);

            job.setMapperClass(FloydWarshallMapper.class);
            job.setReducerClass(FloydWarshallReducer.class);

            job.setOutputKeyClass(Text.class);
            job.setOutputValueClass(Text.class);

            job.getConfiguration().set("mapreduce.output.textoutputformat.separator", ",");

            // Set number of reducers to 1 to ensure a single output file
            job.setNumReduceTasks(1);

            // Set input and output paths for the job
            FileInputFormat.addInputPath(job, new Path(currentInputPath));
            String iterationOutputPath = args[1] + "_" + k;
            FileOutputFormat.setOutputPath(job, new Path(iterationOutputPath));

            // Print debug information
            System.out.println("Starting job for intermediate vertex: " + k);
            System.out.println("Input Path: " + currentInputPath);
            System.out.println("Output Path: " + iterationOutputPath);

            // Delete the output directory if it exists to avoid conflicts
            FileSystem fs = FileSystem.get(conf);
            if (fs.exists(new Path(iterationOutputPath))) {
                System.out.println("Deleting existing output directory: " + iterationOutputPath);
                fs.delete(new Path(iterationOutputPath), true);
            }

            // Run the job and exit if it fails
            if (!job.waitForCompletion(true)) {
                System.exit(1);
            }

            // Verify the output file
            System.out.println("Job for intermediate vertex " + k + " completed.");
            System.out.println("Checking output directory for part file...");

            Path outputFilePath = new Path(iterationOutputPath + "/part-r-00000");
            if (fs.exists(outputFilePath)) {
                System.out.println("Found output file: " + outputFilePath);
            } else {
                System.out.println("Output file not found: " + outputFilePath);
            }

            // Update the current input path to the output file of this iteration
            currentInputPath = iterationOutputPath + "/part-r-00000";
        }

        System.out.println("Floyd-Warshall completed all iterations");
    }
}

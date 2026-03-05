import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

import java.io.*;
import java.net.URI;
import java.util.*;

public class KMeans {

    // Class for when centers are read from file
    public static class Center {
        int id;
        float x;
        float y;

        public Center(int id, float x, float y) {
            this.id = id;
            this.x = x;
            this.y = y;
        }
    }

    public static class KMeansMapper
            extends Mapper<LongWritable, Text, IntWritable, Text> {

        private final List<Center> centers = new ArrayList<>();

        @Override
        protected void setup(Context context) throws IOException {

            // Load small local file into cache
            URI[] cacheFiles = context.getCacheFiles();

            if (cacheFiles == null || cacheFiles.length == 0) {
                throw new IOException("Centers not found in cache.");
            }

            Path cachePath = new Path(cacheFiles[0].getPath());
            File file = new File(cachePath.getName());
            BufferedReader reader = new BufferedReader(new FileReader(file));

            String line;

            while ((line = reader.readLine()) != null) {

                // clusterID \t x,y I believe there is a tab character in between key and value
                String[] parts = line.split("\t");
                int id = Integer.parseInt(parts[0]);
                if (id < 0) {
                    continue;
                }

                String[] coords = parts[1].split(",");
                float x = Float.parseFloat(coords[0]);
                float y = Float.parseFloat(coords[1]);

                centers.add(new Center(id, x, y));
            }
            reader.close();
        }

        @Override
        protected void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {

            // Parse point
            String[] parts = value.toString().split(",");
            int x = Integer.parseInt(parts[0]);
            int y = Integer.parseInt(parts[1]);

            float minDistance = Float.MAX_VALUE;
            int minDistanceClusterID = -1;
            // Loop over all cluster centers
            for (Center c : centers) {

                float dx = x - c.x;
                float dy = y - c.y;

                float distance = dx * dx + dy * dy;

                if (distance < minDistance) {
                    minDistance = distance;
                    minDistanceClusterID = c.id;
                }
            }

            // Add the 1 count for the combiner
            context.write(new IntWritable(minDistanceClusterID), new Text(x + "," + y + ",1"));
        }
    }

    public static class KMeansCombiner
            extends Reducer<IntWritable, Text, IntWritable, Text> {

        // The combiner will just sum each map then send it to the reducer
        @Override
        protected void reduce(IntWritable key, Iterable<Text> values, Context context) throws IOException, InterruptedException {

            float totalX = 0f;
            float totalY = 0f;
            int totalCount = 0;

            for (Text val : values) {

                // x,y,1
                String[] parts = val.toString().split(",");

                float x = Float.parseFloat(parts[0]);
                float y = Float.parseFloat(parts[1]);
                int count = Integer.parseInt(parts[2]);

                totalX += x;
                totalY += y;
                totalCount += count;
            }
            context.write(key, new Text(totalX + "," + totalY + "," + totalCount));
        }
    }

    public static class KMeansReducer
            extends Reducer<IntWritable, Text, IntWritable, Text> {

        private final Map<Integer, Center> centers = new HashMap<>();
        private boolean centersChanged = false;

        @Override
        protected void setup(Context context) throws IOException {

            // Load small local file into cache
            URI[] cacheFiles = context.getCacheFiles();

            if (cacheFiles == null || cacheFiles.length == 0) {
                throw new IOException("Centers not found in cache.");
            }

            Path cachePath = new Path(cacheFiles[0].getPath());
            File file = new File(cachePath.getName());
            BufferedReader reader = new BufferedReader(new FileReader(file));

            String line;

            while ((line = reader.readLine()) != null) {

                // clusterID \t x,y I believe there is a tab character in between key and value
                String[] parts = line.split("\t");
                int id = Integer.parseInt(parts[0]);
                if (id < 0) {
                    continue;
                }

                String[] coords = parts[1].split(",");
                float x = Float.parseFloat(coords[0]);
                float y = Float.parseFloat(coords[1]);

                centers.put(id, new Center(id, x, y));
            }
            reader.close();
        }

        @Override
        protected void reduce(IntWritable key, Iterable<Text> values, Context context) throws IOException, InterruptedException {

            float totalX = 0f;
            float totalY = 0f;
            int totalCount = 0;

            for (Text val : values) {

                // totalX,totalY,totalCount from combiner
                String[] parts = val.toString().split(",");

                float x = Float.parseFloat(parts[0]);
                float y = Float.parseFloat(parts[1]);
                int count = Integer.parseInt(parts[2]);

                totalX += x;
                totalY += y;
                totalCount += count;
            }

            int clusterID = key.get();
            Center oldCenter = centers.get(clusterID);

            if (totalCount == 0) {

                context.write(key, new Text(oldCenter.x + "," + oldCenter.y));
                return;
            }

            float newX = totalX / totalCount;
            float newY = totalY / totalCount;
            float epsilon = 0.01f;

            if (Math.abs(oldCenter.x - newX) > epsilon || Math.abs(oldCenter.y - newY) > epsilon) {

                centersChanged = true;

                context.getCounter("KMeans", "CENTER_CHANGED").increment(1);
            }
            context.write(key, new Text(newX + "," + newY));
        }

        @Override
        protected void cleanup(Context context) throws IOException, InterruptedException {

            context.write(new IntWritable(-1), new Text("CENTERS_CHANGED=" + centersChanged));
        }
    }

    public static void main(String[] args) throws Exception {

        if (args.length != 4) {
            System.err.println("Usage: KMeans <P_input> <output> k");
            System.exit(2);
        }

        int k = Integer.parseInt(args[3]);

        Path inputPath = new Path(args[1]);
        Path outputPathBase = new Path(args[2]);
        String centersPath = outputPathBase + "/centers_0";
        Configuration conf = new Configuration();
        generateInitialCenters(k, centersPath, conf);

        int iteration = 0;
        boolean converged = false;

        while (iteration < 6 && !converged) {

            Job job = Job.getInstance(conf, "KMeans");
            job.setJarByClass(KMeans.class);

            FileInputFormat.addInputPath(job, inputPath);
            String iterationOutput = outputPathBase + "/iter_" + iteration;
            Path outputPath = new Path(iterationOutput);
            // Won't write to file if path exists
            FileSystem.get(conf).delete(outputPath, true);
            FileOutputFormat.setOutputPath(job, outputPath);

            job.addCacheFile(new URI(centersPath));

            job.setMapperClass(KMeans.KMeansMapper.class);
            job.setCombinerClass(KMeansCombiner.class);
            job.setReducerClass(KMeans.KMeansReducer.class);

            job.setMapOutputKeyClass(IntWritable.class);
            job.setMapOutputValueClass(Text.class);
            job.setOutputKeyClass(IntWritable.class);
            job.setOutputValueClass(Text.class);

            job.setNumReduceTasks(1);

            job.waitForCompletion(true);

            long changed = job.getCounters().findCounter("KMeans", "CENTER_CHANGED").getValue();

            if (changed == 0) {
                converged = true; // Will finish after this loop
            }

            centersPath = iterationOutput + "/part-r-00000";

            iteration++;
        }
    }

    public static void generateInitialCenters(int k, String outputPath, Configuration conf) throws IOException {

        FileSystem fs = FileSystem.get(conf);
        Path path = new Path(outputPath);

        if (fs.exists(path)) {
            fs.delete(path, true);
        }

        BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(fs.create(path, true)));

        Random rand = new Random(42);

        for (int i = 0; i < k; i++) {

            float x = rand.nextInt(10_000) + 1;
            float y = rand.nextInt(10_000) + 1;

            writer.write(i + "\t" + x + "," + y + "\n");
        }
        writer.close();
    }
}

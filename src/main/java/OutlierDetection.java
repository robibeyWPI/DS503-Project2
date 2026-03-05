import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

public class OutlierDetection {

    public static class PointsMapper
            extends Mapper<LongWritable, Text, IntWritable, Text> {

        private int r;
        private int numPartitions;
        private static final int COORD_RANGE = 10000;

        private final IntWritable outKey = new IntWritable();
        private final Text outValue = new Text();

        @Override
        protected void setup(Context context) {

            Configuration conf = context.getConfiguration();

            // If the user enters a non-int then it will just error
            r = Integer.parseInt(conf.get("r"));

            if (r <= 0) {
                throw new RuntimeException("r must be positive.");
            }

            numPartitions = COORD_RANGE / r;
        }

        @Override
        public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {

            String[] parts = value.toString().split(",");
            int x = Integer.parseInt(parts[0]);
            int y = Integer.parseInt(parts[1]);

            int cellX = (x - 1) / r;
            int cellY = (y - 1) / r;

            // Divide entire space into only neighbor spaces of the current point of size r
            for (int offsetX = -1; offsetX <= 1; offsetX++) { // space to the left, x itself, space to the right
                for (int offsetY = -1; offsetY <= 1; offsetY++) {

                    int neighborX = cellX + offsetX;
                    int neighborY = cellY + offsetY;

                    if (neighborX >= 0 && neighborY >= 0 && neighborX < numPartitions && neighborY < numPartitions) {

                        int neighborCellID = neighborX * numPartitions + neighborY;

                        outKey.set(neighborCellID);

                        // Original point
                        if (offsetX == 0 && offsetY == 0) {
                            outValue.set("O," + x + "," + y);
                        } else {
                            // Replicated points that belong to neighboring cells
                            outValue.set("R," + x + "," + y);
                        }

                        context.write(outKey, outValue);
                    }
                }
            }
        }
    }

    public static class OutlierReducer
            extends Reducer<IntWritable, Text, Text, Text> {

        private int r;
        private int k;

        @Override
        protected void setup(Context context) {
            Configuration conf = context.getConfiguration();
            r = Integer.parseInt(conf.get("r"));
            k = Integer.parseInt(conf.get("k"));
        }

        @Override
        public void reduce(IntWritable key, Iterable<Text> values, Context context)
                throws IOException, InterruptedException {

            List<int[]> originals = new ArrayList<>();
            List<int[]> allPoints = new ArrayList<>();

            // Separate originals and collect all points
            for (Text val : values) {

                String[] parts = val.toString().split(",");
                // This does count the original point as a neighbor to itself
                int x = Integer.parseInt(parts[1]);
                int y = Integer.parseInt(parts[2]);

                allPoints.add(new int[]{x, y});

                if (parts[0].equals("O")) {
                    originals.add(new int[]{x, y});
                }
            }

            // For each original point, count neighbors
            for (int[] p : originals) {

                int px = p[0];
                int py = p[1];

                int neighborCount = 0;

                for (int[] q : allPoints) {

                    int dx = px - q[0];
                    int dy = py - q[1];

                    // Distance calculation
                    if ((dx * dx) + (dy * dy) <= (r * r)) {
                        neighborCount++;

                        if (neighborCount >= k) {
                            break;
                        }
                    }
                }

                if (neighborCount < k) {
                    context.write(
                            new Text("Outlier"),
                            new Text("(" + px + "," + py + ")")
                    );
                }
            }
        }
    }

    public static void main(String[] args) throws Exception {

        if (args.length != 5) {
            System.err.println("Usage: OutlierDetection <P_input> <output> r k");
            System.exit(2);
        }

        Configuration conf = new Configuration();

        conf.set("r", args[3]);
        conf.set("k", args[4]);

        Job job = Job.getInstance(conf, "Outlier Detection");
        job.setJarByClass(OutlierDetection.class);
        FileInputFormat.addInputPath(job, new Path(args[1]));
        FileOutputFormat.setOutputPath(job, new Path(args[2]));
        job.setMapperClass(PointsMapper.class);
        job.setReducerClass(OutlierReducer.class);
        job.setMapOutputKeyClass(IntWritable.class);
        job.setMapOutputValueClass(Text.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(Text.class);
        job.setNumReduceTasks(16);
        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }
}

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
import org.apache.hadoop.mapreduce.lib.input.MultipleInputs;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

public class SpatialJoin {
    // To avoid losing 15 points we need to assign keys to each point
    // Splitting 10,000 x 10,000 into cells of 1000 x 1000 seems reasonable
    private static final int COORD_RANGE = 10_000;
    private static final int NUM_PARTITIONS = 10;
    private static final int CELL_WIDTH = COORD_RANGE / NUM_PARTITIONS;

    public static class PointsMapper
            extends Mapper<LongWritable, Text, IntWritable, Text> {

        private boolean hasWindow = false;
        private int wx1, wy1, wx2, wy2;

        @Override
        protected void setup(Context context) {

            Configuration conf = context.getConfiguration();

            // Either they all exist, or none do
            if (conf.get("window.x1") != null) {
                hasWindow = true;
                wx1 = Integer.parseInt(conf.get("window.wx1"));
                wy1 = Integer.parseInt(conf.get("window.wy1"));
                wx2 = Integer.parseInt(conf.get("window.wx2"));
                wy2 = Integer.parseInt(conf.get("window.wy2"));
            }
        }

        private final IntWritable outKey = new IntWritable();
        private final Text outValue = new Text();

        @Override
        public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {

            // Format of point,point
            String[] lineParts = value.toString().split(",");

            int x = Integer.parseInt(lineParts[0]);
            int y = Integer.parseInt(lineParts[1]);

            // Skip point if it is outside window bounds
            if (hasWindow) {
                if (x < wx1 || x > wx2 || y < wy1 || y > wy2) {
                    return;
                }
            }

            // We need to calculate which grid cell the point is in
            int cellX = (x - 1) / CELL_WIDTH; // (x - 1) catches boundary conditions
            int cellY = (y - 1) / CELL_WIDTH;
            // Take the point (3209,2841) for example. Integer division makes cellX = 3, cellY = 2
            // Rather than passing 3,2 we just make the key 32.
            int cellID = cellX * NUM_PARTITIONS + cellY;

            outKey.set(cellID);
            outValue.set("P," + x + "," + y); // P,3,2
            context.write(outKey, outValue);
        }
    }

    public static class RectangleMapper
            extends Mapper<LongWritable, Text, IntWritable, Text> {

        private boolean hasWindow = false;
        private int wx1, wy1, wx2, wy2;

        @Override
        protected void setup(Context context) {

            Configuration conf = context.getConfiguration();

            // Either they all exist, or none do
            if (conf.get("window.x1") != null) {
                hasWindow = true;
                wx1 = Integer.parseInt(conf.get("window.wx1"));
                wy1 = Integer.parseInt(conf.get("window.wy1"));
                wx2 = Integer.parseInt(conf.get("window.wx2"));
                wy2 = Integer.parseInt(conf.get("window.wy2"));
            }
        }

        private final IntWritable outKey = new IntWritable();
        private final Text outValue = new Text();

        @Override
        public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {

            // BottomLeftX,BottomLeftY,Height,Width
            String[] lineParts = value.toString().split(",");

            int bottomLeftX = Integer.parseInt(lineParts[0]);
            int bottomLeftY = Integer.parseInt(lineParts[1]);
            int height = Integer.parseInt(lineParts[2]);
            int width = Integer.parseInt(lineParts[3]);

            int maxX = bottomLeftX + width;
            int maxY = bottomLeftY + height;

            // If the rectangle is completely outside the bounds of the window
            if (hasWindow) {
                if (maxX < wx1 || bottomLeftX > wx2 || maxY < wy1 || bottomLeftY > wy2) {
                    return;
                }
            }

            // To determine which cell each point of the rectangle is in
            int minCellX = (bottomLeftX - 1) / CELL_WIDTH;
            int maxCellX = (maxX - 1) / CELL_WIDTH;
            int minCellY = (bottomLeftY - 1) / CELL_WIDTH;
            int maxCellY = (maxY - 1) / CELL_WIDTH;

            // Loop over all the cells the rectangle covers. In x and y.
            // If a rect stretches across multiple cells, the result is continuous. It is not a new rectangle, but cell 2 will get a value of rectangle coordinates outside its cell, which is fine because it stretches into its cell.
            for (int cellX = minCellX; cellX <= maxCellX; cellX++) {
                for (int cellY = minCellY; cellY <= maxCellY; cellY++) {

                    int cellID = cellX * NUM_PARTITIONS + cellY;

                    outKey.set(cellID);
                    outValue.set("R," + bottomLeftX + "," + bottomLeftY + "," + height + "," + width);
                    context.write(outKey, outValue);
                }
            }
        }
    }

    public static class SpatialReducer
            extends Reducer<IntWritable, Text, Text, Text> {

        @Override
        public void reduce(IntWritable key, Iterable<Text> values, Context context) throws IOException, InterruptedException {

            List<int[]> points = new ArrayList<>();
            List<int[]> rects = new ArrayList<>();

            // Separate points and rectangles
            for (Text val : values) {

                String[] parts = val.toString().split(",");
                // x and y are in the same locations
                int x = Integer.parseInt(parts[1]);
                int y = Integer.parseInt(parts[2]);

                if (parts[0].equals("P")) {
                    // P,x,y
                    points.add(new int[]{x, y});
                } else {
                    // R,x,y,h,w
                    int h = Integer.parseInt(parts[3]);
                    int w = Integer.parseInt(parts[4]);
                    rects.add(new int[]{x, y, h, w});
                }
            }

            // Check if point inside rect
            // Loop through each rectangle
            for (int[] rect : rects) {
                int rx = rect[0];
                int ry = rect[1];
                int h = rect[2];
                int w = rect[3];

                int maxX = rx + w;
                int maxY = ry + h;

                // Loop through each point and check if it is within the rectangle's bounds
                for (int[] point : points) {
                    int px = point[0];
                    int py = point[1];

                    // It is possible for a point to exist on multiple rectangle boundaries. In such case, the point is in all sets.
                    if (px >= rx && px <= maxX && py >= ry && py <= maxY) {
                        context.write(
                                new Text("R(" + rx + "," + ry + "," + h + "," + w + ")"),
                                new Text("(" + px + "," + py + ")")
                        );
                    }
                }
            }
        }
    }

    public static void main(String[] args) throws Exception {

        if (args.length != 3 && args.length != 7) {
            System.err.println("Incorrect usage.");
            System.err.println("Without window: SpatialJoin <P_input> <R_input> <output>");
            System.err.println("With window: SpatialJoin <P_input> <R_input> <output> x1 y1 x2 y2");
            System.exit(2);
        }

        Configuration conf = new Configuration();

        if (args.length == 7) {
            conf.set("window.x1", args[3]);
            conf.set("window.y1", args[4]);
            conf.set("window.x2", args[5]);
            conf.set("window.y2", args[6]);
        }

        Job job = Job.getInstance(conf, "Spatial Join");
        job.setJarByClass(SpatialJoin.class);
        MultipleInputs.addInputPath(job, new Path(args[0]), TextInputFormat.class, PointsMapper.class);
        MultipleInputs.addInputPath(job, new Path(args[1]), TextInputFormat.class, RectangleMapper.class);
        job.setReducerClass(SpatialReducer.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(Text.class);
        job.setNumReduceTasks(8);
        FileOutputFormat.setOutputPath(job, new Path(args[2]));
        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }
}

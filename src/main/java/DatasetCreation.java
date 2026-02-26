import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.concurrent.ThreadLocalRandom;

public class DatasetCreation {
    // Space bounds
    private static final int MIN_COORD_BOUND = 1;
    private static final int MAX_COORD_BOUND = 10_000;

    // Rectangle size bounds
    private static final int MAX_HEIGHT = 20;
    private static final int MAX_WIDTH = 7;

    // Number of records
    private static final int NUM_POINTS = 10_000_000;
    private static final int NUM_RECTS = 7_000_000;

    public static void main(String[] args) throws IOException {
        generatePoints("P.txt", NUM_POINTS);
        generateRects("R.txt", NUM_RECTS);
        System.out.println("Generated datasets.");
    }

    private static void generatePoints(String fileName, int count) throws IOException {
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(fileName))) {
            for (int i = 0; i < count; i ++) {
                int x = ThreadLocalRandom.current().nextInt(MIN_COORD_BOUND, MAX_COORD_BOUND + 1);
                int y = ThreadLocalRandom.current().nextInt(MIN_COORD_BOUND, MAX_COORD_BOUND + 1);

                writer.write(x + "," + y);
                writer.newLine();
            }
        }
    }

    private static void generateRects(String fileName, int count) throws IOException {
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(fileName))) {
            for (int i = 0; i < count; i ++) {
                int bottomLeftX = ThreadLocalRandom.current().nextInt(MIN_COORD_BOUND, MAX_COORD_BOUND + 1);
                int bottomLeftY = ThreadLocalRandom.current().nextInt(MIN_COORD_BOUND, MAX_COORD_BOUND + 1);

                int height = ThreadLocalRandom.current().nextInt(1, MAX_HEIGHT + 1);
                int width = ThreadLocalRandom.current().nextInt(1, MAX_WIDTH + 1);

                writer.write(bottomLeftX + "," + bottomLeftY + "," + height + "," + width);
                writer.newLine();
            }
        }
    }
}

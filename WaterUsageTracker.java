import java.io.*;
import java.time.LocalDate;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.Scanner;

class Bill {
    private double amount;
    private String message;

    public Bill(double amount, String message) {
        this.amount = amount;
        this.message = message;
    }

    public double getAmount() {
        return amount;
    }

    public String getMessage() {
        return message;
    }
}

public class WaterUsageTracker {
    private ArrayList<Integer> dailyUsage = new ArrayList<>();
    private ArrayList<Integer> monthlyUsage = new ArrayList<>();
    private static final String FILE_NAME = "monthly_usage.txt";
    private static final int DAYS_IN_MONTH = 28;

    public WaterUsageTracker() {
        loadMonthlyUsage();
    }

    public void addDailyUsage(int liters) {
        dailyUsage.add(liters);
    }

    public int getTotalUsage() {
        int total = 0;
        for (int usage : dailyUsage) {
            total += usage;
        }
        return total;
    }

    public void addMonthlyUsage(int liters) {
        monthlyUsage.add(liters);
        saveMonthlyUsage();
    }

    public int getLastMonthUsage() {
        if (monthlyUsage.size() < 2) {
            return 0;
        }
        return monthlyUsage.get(monthlyUsage.size() - 2);
    }

    public Bill calculateBill() {
        int totalUsage = getTotalUsage();
        int lastMonthUsage = getLastMonthUsage();
        double bill = totalUsage * 0.041; 
        String alertMessage = "";

        if (totalUsage > lastMonthUsage) {
            double increasePercentage = ((double)(totalUsage - lastMonthUsage) / lastMonthUsage) * 100;
            if (increasePercentage > 50) {
                bill *= 1.20;
                alertMessage = "Alert: Your water usage has increased by more than 50%. Your bill includes a 20% increase.";
            } else if (increasePercentage > 20) {
                bill *= 1.15;
                alertMessage = "Alert: Your water usage has increased by more than 20%. Your bill includes a 15% increase.";
            } else {
                bill *= 1.10;
                alertMessage = "Alert: Your water usage has increased. Your bill includes a 10% increase.";
            }
        }

        return new Bill(bill, alertMessage);
    }

    public void printWaterSavingTips(double increasePercentage) {
        System.out.println("Water Saving Tips:");
        if (increasePercentage > 50) {
            System.out.println("1. Install a water-efficient showerhead.");
            System.out.println("2. Reduce shower time.");
            System.out.println("3. Use a broom instead of a hose to clean driveways.");
        } else if (increasePercentage > 20) {
            System.out.println("1. Fix leaks promptly.");
            System.out.println("2. Use water-saving fixtures.");
            System.out.println("3. Turn off the tap while brushing your teeth.");
        } else {
            System.out.println("1. Collect rainwater for gardening.");
            System.out.println("2. Water plants during the cooler parts of the day.");
            System.out.println("3. Use a bucket to wash your car instead of a hose.");
        }
    }

    private void saveMonthlyUsage() {
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(FILE_NAME))) {
            for (int usage : monthlyUsage) {
                writer.write(usage + "\n");
            }
        } catch (IOException e) {
            System.out.println("Error saving monthly usage: " + e.getMessage());
        }
    }

    private void loadMonthlyUsage() {
        try (BufferedReader reader = new BufferedReader(new FileReader(FILE_NAME))) {
            String line;
            while ((line = reader.readLine()) != null) {
                monthlyUsage.add(Integer.parseInt(line));
            }
        } catch (IOException e) {
            System.out.println("Error loading monthly usage: " + e.getMessage());
        }
    }

    private static String getCurrentMonthName() {
        LocalDate currentDate = LocalDate.now();
        return currentDate.format(DateTimeFormatter.ofPattern("MMMM"));
    }

    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        WaterUsageTracker tracker = new WaterUsageTracker();
        String currentMonth = getCurrentMonthName();

        try {
            System.out.println("Enter daily water usage in liters for " + currentMonth + ":");
            for (int day = 1; day <= DAYS_IN_MONTH; day++) {
                System.out.print("Day-" + day + ": Usage: ");
                int usage = scanner.nextInt();
                tracker.addDailyUsage(usage);
            }

            int totalUsage = tracker.getTotalUsage();
            tracker.addMonthlyUsage(totalUsage);
            Bill bill = tracker.calculateBill();

            System.out.println("\nTotal water usage for " + currentMonth + ": " + totalUsage + " liters");
            System.out.println("Total bill: INR " + bill.getAmount());
            if (!bill.getMessage().isEmpty()) {
                System.out.println(bill.getMessage());
            }

            int lastMonthUsage = tracker.getLastMonthUsage();
            double increasePercentage = lastMonthUsage > 0 ? ((double)(totalUsage - lastMonthUsage) / lastMonthUsage) * 100 : 0;
            tracker.printWaterSavingTips(increasePercentage);
        } finally {
            scanner.close();
        }
    }
}
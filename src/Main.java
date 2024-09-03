import javax.sql.rowset.spi.XmlReader;
import java.beans.XMLDecoder;
import java.io.*;
import java.text.ParseException;
import java.util.*;
import java.util.regex.Pattern;

public class Main {

    public static void main(String[] args) {
        pearlTest();
    }

    public static void pearlTest() {
        var file = new File("res/test.bif");
        try {
            var bifParser = new BifParser(file);
            BayesianNetwork network = bifParser.parse();
            var query = Set.of("BURGLARY");
            var evidence = Map.<String, String>of();
            System.out.println(Arrays.toString(network.askEliminationDist(query, evidence).getProbs().getNums()));
        } catch (IOException e) {
            throw new RuntimeException(e);
        } catch (ParseException e) {
            throw new RuntimeException(e);
        }
    }

    public static void win95test() {
        var file = new File("res/win95pts.bif");
        try {
            var bifParser = new BifParser(file);
            BayesianNetwork network = bifParser.parse();
            var query = Set.of("Problem1", "Problem2", "Problem3", "Problem4", "Problem5", "Problem6");
            var evidence = Map.<String, String>of();
            System.out.println(Arrays.toString(network.askEliminationDist(query, evidence).getProbs().getNums()));
        } catch (IOException e) {
            throw new RuntimeException(e);
        } catch (ParseException e) {
            throw new RuntimeException(e);
        }
    }
}

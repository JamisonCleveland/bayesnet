import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

class BayesianNetworkTest {
    BayesianNetwork network;

    @BeforeEach
    void setUp() {
        var dependencyGraph = new ArrayList<List<Integer>>();
        var varNames = new ArrayList<String>();
        var vars = new ArrayList<Variable>();
        var tables = new ArrayList<Table>();

        int varCount = 0;

        int burglaryVarId = varCount++;
        var burglaryTable = mkBoolTable(List.of(burglaryVarId), List.of(0.001));

        dependencyGraph.add(List.of());
        varNames.add("Burglary");
        vars.add(new Variable(List.of("False", "True")));
        tables.add(burglaryTable);

        int earthquakeVarId = varCount++;
        var earthquakeTable = mkBoolTable(List.of(earthquakeVarId), List.of(0.002));

        dependencyGraph.add(List.of());
        varNames.add("Earthquake");
        vars.add(new Variable(List.of("False", "True")));
        tables.add(earthquakeTable);

        int alarmVarId = varCount++;
        var alarmTable = mkBoolTable(List.of(burglaryVarId, earthquakeVarId, alarmVarId), List.of(0.001, 0.29, 0.94, 0.95));

        dependencyGraph.add(List.of(burglaryVarId, earthquakeVarId));
        varNames.add("Alarm");
        vars.add(new Variable(List.of("False", "True")));
        tables.add(alarmTable);

        int johnCallsVarId = varCount++;
        var johnCallsTable = mkBoolTable(List.of(alarmVarId, johnCallsVarId), List.of(0.05, 0.9));

        dependencyGraph.add(List.of(alarmVarId));
        varNames.add("JohnCalls");
        vars.add(new Variable(List.of("False", "True")));
        tables.add(johnCallsTable);

        int maryCallsVarId = varCount++;
        var maryCallsTable = mkBoolTable(List.of(alarmVarId, maryCallsVarId), List.of(0.01, 0.7));

        dependencyGraph.add(List.of(alarmVarId));
        varNames.add("MaryCalls");
        vars.add(new Variable(List.of("False", "True")));
        tables.add(maryCallsTable);

        network = new BayesianNetwork(dependencyGraph, vars, varNames, tables);
    }

    private static Table mkBoolTable(List<Integer> varIds, List<Double> qProbs) {
        int totalSize = 1 << varIds.size();
        assert totalSize == 2 * qProbs.size() : "size and number of q values must be equal";
        var sizes = Collections.nCopies(varIds.size(), 2);

        double[] nums = new double[totalSize];
        for (int i = 0; i < qProbs.size(); i++) {
            double q = qProbs.get(i);
            nums[2 * i] = 1.0 - q;
            nums[2 * i + 1] = q;
        }
        var probs = new NDArray(sizes, nums);

        return new Table(varIds, probs);
    }

    @Test
    void askEliminationDist() {
        var query = Set.of("Burglary");
        var evidence = Map.of("JohnCalls", "True", "MaryCalls", "True");
        var expected = new double[]{0.716, 0.284};

        var res = network.askEliminationDist(query, evidence);

        assertArrayEquals(expected, res.getProbs().getNums(), 1e-3);
    }

    @Test
    void askEliminationMode() {
        var query = Set.of("Burglary", "Earthquake", "Alarm");
        var evidence = Map.of("JohnCalls", "True", "MaryCalls", "True");

        var expected = Map.of("Burglary", "False", "Earthquake", "False", "Alarm", "True");
        var res = network.askEliminationMode(query, evidence);
        assertEquals(expected, res);
    }
}
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

class BayesianNetworkTest {
    BayesianNetwork network;

    @BeforeEach
    void setUp() {
        var graph = new ArrayList<List<Integer>>();
        var varNames = new ArrayList<String>();
        var vars = new ArrayList<Variable>();
        var varIdx = new HashMap<String, Integer>();
        var tables = new ArrayList<Table>();

        int varCount = 0;

        int burglaryVarId = varCount++;
        var burglaryDeps = List.<Integer>of();
        var burglaryName = "Burglary";
        var burglaryVar = new Variable(List.of("False", "True"));
        var burglaryTable = mkBoolTable(List.of(burglaryVarId), List.of(0.001));

        graph.add(burglaryDeps);
        varNames.add(burglaryName);
        vars.add(burglaryVar);
        tables.add(burglaryTable);
        varIdx.put(burglaryName, burglaryVarId);

        int earthquakeVarId = varCount++;
        var earthquakeDeps = List.<Integer>of();
        var earthquakeName = "Earthquake";
        var earthquakeVar = new Variable(List.of("False", "True"));
        var earthquakeTable = mkBoolTable(List.of(earthquakeVarId), List.of(0.002));

        graph.add(earthquakeDeps);
        varNames.add(earthquakeName);
        vars.add(earthquakeVar);
        tables.add(earthquakeTable);
        varIdx.put(earthquakeName, earthquakeVarId);

        int alarmVarId = varCount++;
        var alarmDeps = List.of(burglaryVarId, earthquakeVarId);
        var alarmName = "Alarm";
        var alarmVar = new Variable(List.of("False", "True"));
        var alarmTable = mkBoolTable(List.of(burglaryVarId, earthquakeVarId, alarmVarId), List.of(0.001, 0.29, 0.94, 0.95));

        graph.add(alarmDeps);
        varNames.add(alarmName);
        vars.add(alarmVar);
        tables.add(alarmTable);
        varIdx.put(alarmName, alarmVarId);

        int johnCallsVarId = varCount++;
        var johnCallsDeps = List.of(alarmVarId);
        var johnCallsName = "JohnCalls";
        var johnCallsVar = new Variable(List.of("False", "True"));
        var johnCallsTable = mkBoolTable(List.of(alarmVarId, johnCallsVarId), List.of(0.05, 0.9));

        graph.add(johnCallsDeps);
        varNames.add(johnCallsName);
        vars.add(johnCallsVar);
        tables.add(johnCallsTable);
        varIdx.put(johnCallsName, johnCallsVarId);

        int maryCallsVarId = varCount++;
        var maryCallsDeps = new ArrayList<Integer>();
        maryCallsDeps.add(alarmVarId);
        var maryCallsName = "MaryCalls";
        var maryCallsVar = new Variable(List.of("False", "True"));
        var maryCallsTable = mkBoolTable(List.of(alarmVarId, maryCallsVarId), List.of(0.01, 0.7));

        graph.add(maryCallsDeps);
        varNames.add(maryCallsName);
        vars.add(maryCallsVar);
        tables.add(maryCallsTable);
        varIdx.put(maryCallsName, 4);

        network = new BayesianNetwork(graph,vars,varIdx,varNames,tables);
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
    void ask() {
        var query = List.of("Burglary");
        var evidence = Map.of("JohnCalls", "True", "MaryCalls", "True");
        var expected = new double[]{0.716, 0.284};

        var res = network.ask(query, evidence);

        assertArrayEquals(expected, res.getProbs().getNums(), 1e-3);
    }
}
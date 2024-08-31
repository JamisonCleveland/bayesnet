import java.util.*;

public class Main {
    public static void main(String[] args) {
        // TODO: extract into unit tests
        // TODO: maybe network factory?
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

        var bayesnet = new BayesianNetwork(graph,vars,varIdx,varNames,tables);

        var query = List.of("Burglary");
        var evidence = Map.of("JohnCalls", "True", "MaryCalls", "True");

        // manually ran example
        var a1 = maryCallsTable.condition(maryCallsVarId, 1);
        var a2 = johnCallsTable.condition(johnCallsVarId, 1);
        var a4 = alarmTable.product(a1).product(a2);
        var a5 = a4.marginalize(alarmVarId);
        var a6 = earthquakeTable.product(a5);
        var a7 = a6.marginalize(earthquakeVarId);
        var a8 = a7.product(burglaryTable);
        a8.normalize();

        System.out.println(Arrays.toString(a8.getProbs().getNums()));
        System.out.println(a8.getVarIds());

        var res = bayesnet.ask(query, evidence);
        System.out.println(Arrays.toString(res.getProbs().getNums()));

        System.out.println("Hello world!");
    }

    public static Table mkBoolTable(List<Integer> varIds, List<Double> qProbs) {
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
}

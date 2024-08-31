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

// Maybe a Network Builder/Factory?
class BayesianNetwork {
    private List<List<Integer>> graph;
    private List<Variable> vars;
    private List<String> varNames;
    private Map<String, Integer> varIds;
    private List<Table> tables;

    public BayesianNetwork(List<List<Integer>> graph, List<Variable> vars, Map<String, Integer> varIds, List<String> varNames, List<Table> tables) {
        this.graph = graph;
        this.vars = vars;
        this.varIds = varIds;
        this.varNames = varNames;
        this.tables = tables;
    }

    private List<Integer> topSort() {
        var order = new ArrayList<Integer>();
        var permMark = new HashSet<Integer>();
        for (int node = 0; node < graph.size(); node++) {
            if (permMark.contains(node)) continue;
            dfs(node, order, permMark, new HashSet<Integer>());
        }
        return order;
    }

    private void dfs(int n, List<Integer> order, HashSet<Integer> permMark, HashSet<Integer> tempMark) {
        if (permMark.contains(n)) {
            return;
        }
        assert !tempMark.contains(n) : "graph must be acyclic";

        tempMark.add(n);
        for (var m : graph.get(n)) {
            dfs(m, order, permMark, tempMark);
        }
        permMark.add(n);
        order.add(n);
    }

    public Table ask(List<String> query, Map<String, String> evidence) {
        var order = topSort();
        Collections.reverse(order);

        // condition factors
        var factors = new ArrayList<>(tables);
        for (var entry : evidence.entrySet()) {
            System.out.println("Conditioning: " + entry.getKey() + "=" + entry.getValue());
            int varId = varIds.get(entry.getKey());
            int valId = vars.get(varId).getValNames().indexOf(entry.getValue());

            for (int i = 0; i < factors.size(); i++) {
                var factor = factors.get(i);
                if (!factor.getVarIds().contains(varId)) continue;
                factors.set(i, factor.condition(varId, valId));
            }
        }

        // Eliminate variables
        for (var varId : order) {
            var varName = varNames.get(varId);
            if (query.contains(varName) || evidence.containsKey(varName)) continue;
            System.out.println("Eliminating: " + varName);

            // product of all factors containing n
            int i = 0;
            var prodMaybe = factors.stream()
                    .filter(factor -> factor.getVarIds().contains(varId))
                    .reduce(Table::product);
            assert prodMaybe.isPresent();
            var prod = prodMaybe.get();

            factors.removeIf(factor -> factor.getVarIds().contains(varId));

            // add marginalized to factors
            factors.add(prod.marginalize(varId));
        }

        var resMaybe = factors.stream().reduce(Table::product);
        assert resMaybe.isPresent();
        var res = resMaybe.get();
        res.normalize();
        return res;
    }
}

class Variable {
    private List<String> valNames;
    private Map<String, Integer> valIdx;

    public Variable(List<String> valNames) {
        this.valNames = valNames;
        this.valIdx = new HashMap<>();
        for (int i = 0; i < valNames.size(); i++) {
            String val = valNames.get(i);
            this.valIdx.put(val, i);
        }
    }

    public List<String> getValNames() {
        return valNames;
    }
}

class Table {
    private List<Integer> varIds;
    private NDArray probs;

    public Table(List<Integer> varIds, NDArray probs) {
        this.probs = probs;
        this.varIds = varIds;
    }

    public List<Integer> getVarIds() {
        return varIds;
    }

    public NDArray getProbs() {
        return probs;
    }

    public Table condition(int varId, int valId) {
        int axis = this.varIds.indexOf(varId);
        var probs = this.probs.filter(axis, valId);
        var varIds = new ArrayList<>(this.varIds);
        varIds.remove(axis);

        return new Table(varIds, probs);
    }

    public Table marginalize(int varId) {
        int axis = this.varIds.indexOf(varId);
        var probs = this.probs.sumOver(axis);
        var varIds = new ArrayList<>(this.varIds);
        varIds.remove(axis);

        return new Table(varIds, probs);
    }
    public void normalize() {
        probs.normalize();
    }

    public Table product(Table that) {
        var union = new ArrayList<Integer>();
        var thisAxes = new ArrayList<Integer>();
        var thatAxes = new ArrayList<Integer>();
        for (int thisVarId : this.varIds) {
            if (that.varIds.contains(thisVarId)) {
                union.add(thisVarId);
                thisAxes.add(this.varIds.indexOf(thisVarId));
                thatAxes.add(that.varIds.indexOf(thisVarId));
            }
        }

        var newVarIds = new ArrayList<Integer>();
        for (int thisVarId : this.varIds) {
            if (union.contains(thisVarId)) continue;
            newVarIds.add(thisVarId);
        }
        for (int thatVarId : that.varIds) {
            if (union.contains(thatVarId)) continue;
            newVarIds.add(thatVarId);
        }
        newVarIds.addAll(union);

        var newProbs = this.probs.join(that.probs, thisAxes, thatAxes);

        return new Table(newVarIds, newProbs);
    }
}


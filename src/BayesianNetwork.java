import java.util.*;

// Maybe a Network Builder/Factory?
public class BayesianNetwork {
    private final List<List<Integer>> dependencyGraph;
    private final List<List<Integer>> graph;
    private final List<Variable> vars;
    private final List<String> varNames;
    private final Map<String, Integer> varIds;
    private final List<Table> tables;

    public BayesianNetwork(List<List<Integer>> dependencyGraph, List<Variable> vars, List<String> varNames, List<Table> tables) {
        this.dependencyGraph = dependencyGraph;
        this.graph = new ArrayList<>();
        for (int i = 0; i < dependencyGraph.size(); i++) {
            this.graph.add(new ArrayList<>());
        }
        for (int i = 0; i < dependencyGraph.size(); i++) {
            var neighbors = dependencyGraph.get(i);
            for (int j : neighbors) {
                this.graph.get(j).add(i);
            }
        }

        this.vars = vars;
        this.varNames = varNames;
        varIds = new HashMap<>();
        for (int i = 0; i < varNames.size(); i++) {
            varIds.put(varNames.get(i), i);
        }

        this.tables = tables;
    }

    private static List<Integer> topSort(List<List<Integer>> graph) {
        var order = new ArrayList<Integer>();
        var permMark = new HashSet<Integer>();
        for (int node = 0; node < graph.size(); node++) {
            if (permMark.contains(node)) continue;
            dfs(graph, node, order, permMark, new HashSet<>());
        }
        return order;
    }

    private static void dfs(List<List<Integer>> graph, int n, List<Integer> order, HashSet<Integer> permMark, HashSet<Integer> tempMark) {
        if (permMark.contains(n)) {
            return;
        }
        assert !tempMark.contains(n) : "graph must be acyclic";

        tempMark.add(n);
        for (var m : graph.get(n)) {
            dfs(graph, m, order, permMark, tempMark);
        }
        permMark.add(n);
        order.add(n);
    }

    public Table askEliminationDist(Set<String> query, Map<String, String> evidence) {
        var order = topSort(graph);

        // condition factors
        var factors = new ArrayList<>(tables);
        for (var entry : evidence.entrySet()) {
            int varId = varIds.get(entry.getKey());
            int valId = vars.get(varId).getValIds().get(entry.getValue());

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

            // product of all factors containing n
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

    public Map<String, String> askEliminationMode(Set<String> query, Map<String, String> evidence) {
        var order = topSort(graph);

        // condition factors
        var factors = new ArrayList<>(tables);
        for (var entry : evidence.entrySet()) {
            int varId = varIds.get(entry.getKey());
            int valId = vars.get(varId).getValIds().get(entry.getValue());

            for (int i = 0; i < factors.size(); i++) {
                var factor = factors.get(i);
                if (!factor.getVarIds().contains(varId)) continue;
                factors.set(i, factor.condition(varId, valId));
            }
        }

        // Eliminate variables
        var res = new HashMap<String, String>();
        for (var varId : order) {
            var varName = varNames.get(varId);
            if (query.contains(varName) || evidence.containsKey(varName)) continue;

            // product of all factors containing n
            var prodMaybe = factors.stream()
                    .filter(factor -> factor.getVarIds().contains(varId))
                    .reduce(Table::product);
            assert prodMaybe.isPresent();
            var prod = prodMaybe.get();

            factors.removeIf(factor -> factor.getVarIds().contains(varId));

            int valId = prod.mode(varId);
            factors.add(prod.condition(varId, valId));

            var var = vars.get(varId);
            res.put(varName, var.getValNames().get(valId));
        }

        var factorMaybe = factors.stream().reduce(Table::product);
        assert factorMaybe.isPresent();
        var factor = factorMaybe.get();

        List<Integer> valIds = factor.mode();
        for (int i = 0; i < factor.getVarIds().size(); i++) {
            int varId = factor.getVarIds().get(i);
            String varName = varNames.get(varId);
            var var = vars.get(varId);

            int valId = valIds.get(i);
            String valName = var.getValNames().get(valId);

            res.put(varName, valName);
        }

        return res;
    }
}

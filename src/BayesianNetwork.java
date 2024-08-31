import java.util.*;

// Maybe a Network Builder/Factory?
class BayesianNetwork {
    private final List<List<Integer>> graph;
    private final List<Variable> vars;
    private final List<String> varNames;
    private final Map<String, Integer> varIds;
    private final List<Table> tables;

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
            dfs(node, order, permMark, new HashSet<>());
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
}

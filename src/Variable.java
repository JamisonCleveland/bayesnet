import java.util.HashMap;
import java.util.List;
import java.util.Map;

class Variable {
    private List<String> valNames;
    private Map<String, Integer> valIds;

    public Variable(List<String> valNames) {
        this.valNames = valNames;
        this.valIds = new HashMap<>();
        for (int i = 0; i < valNames.size(); i++) {
            String val = valNames.get(i);
            this.valIds.put(val, i);
        }
    }

    public List<String> getValNames() {
        return valNames;
    }

    public Map<String, Integer> getValIds() {
        return valIds;
    }
}

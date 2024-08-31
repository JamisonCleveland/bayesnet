import java.util.ArrayList;
import java.util.List;

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

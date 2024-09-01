import java.util.*;
import java.util.stream.Collectors;

class NDArray {
    private final double[] nums;
    private final List<Integer> shape;

    public NDArray(List<Integer> shape, double[] nums) {
        this.nums = nums;
        this.shape = shape;
    }

    public double[] getNums() {
        return nums;
    }

    public List<Integer> getShape() {
        return shape;
    }

    public NDArray filter(int axis, int idx) {
        if (axis < 0 || axis >= this.shape.size()) {
            throw new IllegalArgumentException("Axis out of bounds");
        }
        if (idx < 0 || idx >= this.shape.get(axis)) {
            throw new IllegalArgumentException("Index out of bounds");
        }

        List<Integer> newShape = new ArrayList<>(this.shape);
        newShape.remove(axis);

        int newSize = newShape.stream().reduce(1, (a, b) -> a * b);
        double[] newNums = new double[newSize];

        int stride = 1;
        for (int i = axis + 1; i < this.shape.size(); i++) {
            stride *= this.shape.get(i);
        }

        int offset = idx * stride;

        for (int i = 0; i < newSize; i++) {
            int oldIndex = (i / stride) * this.shape.get(axis) * stride + offset + (i % stride);
            newNums[i] = this.nums[oldIndex];
        }

        return new NDArray(newShape, newNums);
    }

    public NDArray sumOver(int axis) {
        if (axis < 0 || axis >= this.shape.size()) {
            throw new IllegalArgumentException("Axis out of bounds");
        }

        List<Integer> newShape = new ArrayList<>(this.shape);
        newShape.remove(axis);

        int newSize = newShape.stream().reduce(1, (a, b) -> a * b);
        double[] newNums = new double[newSize];

        int stride = 1;
        for (int i = axis + 1; i < this.shape.size(); i++) {
            stride *= this.shape.get(i);
        }

        int axisSize = this.shape.get(axis);

        for (int i = 0; i < newSize; i++) {
            double sum = 0;
            for (int j = 0; j < axisSize; j++) {
                int offset = j * stride;
                int oldIndex = (i / stride) * this.shape.get(axis) * stride + offset + (i % stride);
                sum += this.nums[oldIndex];
            }
            newNums[i] = sum;
        }

        return new NDArray(newShape, newNums);
    }

    public void normalize() {
        double sum = Arrays.stream(nums).sum();
        for (int i = 0; i < nums.length; i++) {
            nums[i] /= sum;
        }
    }

    public NDArray join(NDArray that, List<Integer> thisAxes, List<Integer> thatAxes) {
        assert thisAxes.size() == thatAxes.size();
        for (int i = 0; i < thisAxes.size(); i++) {
            assert Objects.equals(this.shape.get(thisAxes.get(i)), that.shape.get(thatAxes.get(i)));
        }

        List<Integer> resultShape = new ArrayList<>();
        boolean[] joinedAxesThis = new boolean[this.shape.size()];
        boolean[] joinedAxesThat = new boolean[that.shape.size()];

        for (int i = 0; i < thisAxes.size(); i++) {
            joinedAxesThis[thisAxes.get(i)] = true;
            joinedAxesThat[thatAxes.get(i)] = true;
        }

        for (int i = 0; i < this.shape.size(); i++) {
            if (!joinedAxesThis[i])
                resultShape.add(this.shape.get(i));
        }
        for (int i = 0; i < that.shape.size(); i++) {
            if (!joinedAxesThat[i])
                resultShape.add(that.shape.get(i));
        }
        for (Integer thisAx : thisAxes) {
            resultShape.add(this.shape.get(thisAx));
        }

        int resultSize = resultShape.stream().reduce(1, (a, b) -> a * b);
        double[] resultNums = new double[resultSize];

        int[] thisStrides = computeStrides(this.shape);
        int[] thatStrides = computeStrides(that.shape);
        int[] resultStrides = computeStrides(resultShape);

        // Iterate over the resulting array and populate values
        for (int resultIndex = 0; resultIndex < resultSize; resultIndex++) {
            int[] resultMultiIndex = toMultiIndex(resultIndex, resultStrides);

            int thisIndex = 0;
            int thatIndex = 0;
            int resultMultiIndexPos = 0;

            for (int i = 0; i < this.shape.size(); i++) {
                if (!joinedAxesThis[i]) {
                    thisIndex += resultMultiIndex[resultMultiIndexPos] * thisStrides[i];
                    resultMultiIndexPos++;
                }
            }

            for (int i = 0; i < that.shape.size(); i++) {
                if (!joinedAxesThat[i]) {
                    thatIndex += resultMultiIndex[resultMultiIndexPos] * thatStrides[i];
                    resultMultiIndexPos++;
                }
            }

            for (int i = 0; i < thisAxes.size(); i++) {
                int joinIndex = resultMultiIndex[resultMultiIndexPos++];
                thisIndex += joinIndex * thisStrides[thisAxes.get(i)];
                thatIndex += joinIndex * thatStrides[thatAxes.get(i)];
            }

            resultNums[resultIndex] = this.nums[thisIndex] * that.nums[thatIndex];
        }

        return new NDArray(resultShape, resultNums);
    }

    private static int[] computeStrides(List<Integer> shape) {
        int[] strides = new int[shape.size()];
        int stride = 1;
        for (int i = shape.size() - 1; i >= 0; i--) {
            strides[i] = stride;
            stride *= shape.get(i);
        }
        return strides;
    }

    private static int[] toMultiIndex(int index, int[] strides) {
        int[] multiIndex = new int[strides.length];
        for (int i = 0; i < strides.length; i++) {
            multiIndex[i] = index / strides[i];
            index = index % strides[i];
        }
        return multiIndex;
    }

    private static int toLinearIndex(int[] multiIndex, int[] strides) {
        int linearIndex = 0;
        for (int i = 0; i < multiIndex.length; i++) {
            linearIndex += strides[i] * multiIndex[i];
        }
        return linearIndex;
    }

    public int argmax(int axis) {
        int stride = 1;
        for (int i = axis + 1; i < this.shape.size(); i++) {
            stride *= this.shape.get(i);
        }
        int argmax = -1;
        double max = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < this.shape.get(axis); i++) {
            double x = this.nums[stride * i];
            if (x > max) {
                argmax = i;
                max = x;
            }
        }
        assert argmax >= 0;
        return argmax;
    }

    public List<Integer> argmax() {
        int argmax = -1;
        double max = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < this.nums.length; i++) {
            double x = this.nums[i];
            if (x > max) {
                argmax = i;
                max = x;
            }
        }
        assert argmax >= 0;
        var strides = computeStrides(this.shape);
        var multiIndex = toMultiIndex(argmax, strides);
        return Arrays.stream(multiIndex).boxed().toList();
    }
}

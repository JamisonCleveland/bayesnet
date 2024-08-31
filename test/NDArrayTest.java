import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class NDArrayTest {

    private NDArray array;
    private NDArray array2;

    @BeforeEach
    void setUp() {
        double[] nums1 = {1, 3, 2, 1};
        List<Integer> shape1 = Arrays.asList(2, 2);
        array = new NDArray(shape1, nums1);

        double[] nums2 = {4, 3, 1, 2};
        List<Integer> shape2 = Arrays.asList(2, 2);
        array2 = new NDArray(shape2, nums2);
    }

    @Test
    void filter() {
        NDArray filteredArray = array.filter(0, 0);

        double[] expectedNums = {1, 3};
        List<Integer> expectedShape = List.of(2);

        assertArrayEquals(expectedNums, filteredArray.getNums(), "Filtered array values do not match");
        assertEquals(expectedShape, filteredArray.getShape(), "Filtered array shape does not match");
    }

    @Test
    void join() {
        NDArray joinedArray = array.join(array2, List.of(1), List.of(0));

        double[] expectedNums = {4, 3, 3, 6, 8, 1, 6, 2};
        List<Integer> expectedShape = Arrays.asList(2, 2, 2);

        assertArrayEquals(expectedNums, joinedArray.getNums(), "Joined array values do not match");
        assertEquals(expectedShape, joinedArray.getShape(), "Joined array shape does not match");
    }

    @Test
    void sumOver() {
        NDArray summedArray = array.sumOver(0);

        double[] expectedNums = {3, 4};
        List<Integer> expectedShape = List.of(2);

        assertArrayEquals(expectedNums, summedArray.getNums(), "Summed array values do not match");
        assertEquals(expectedShape, summedArray.getShape(), "Summed array shape does not match");
    }
}
import java.io.*;
import java.text.ParseException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class BifParser {
    char[] buf;
    int pos = 0;
    public BifParser(File source) throws IOException {
        int len = (int) source.length();
        buf = new char[len];
        var a = new FileReader(source);
        a.read(buf, 0, len);
        a.close();
    }

    // TODO: needs a lot of refactoring
    public BayesianNetwork parse() throws ParseException {
        // :(
        // Parse through the file
        var variableNames = new ArrayList<String>();
        var variables = new ArrayList<Variable>();
        var tableVariableNames = new ArrayList<String>();
        var tableConditionalVariableNames = new ArrayList<List<String>>();
        var tableConditionalProbs = new ArrayList<List<List<Double>>>();
        var tableConditionalValues = new ArrayList<List<List<String>>>();

        while (pos < buf.length) {
            if (accept("network")) {
                identifier();
                expect("{");
                expect("}");
            } else if (accept("variable")) {
                var variableName = identifier();
                expect("{");
                expect("type");
                expect("discrete");
                expect("[");
                integer(); // we ignore it 'cause we don't need it
                expect("]");
                expect("{");
                var valueNames = new ArrayList<String>();
                do {
                    var valueName = identifier();
                    valueNames.add(valueName);
                } while (accept(","));
                expect("}");
                expect(";");
                expect("}");

                variableNames.add(variableName);
                variables.add(new Variable(valueNames));
            } else if (accept("probability")) {
                expect("(");
                var variableName = identifier();
                var conditionalVariableNames = new ArrayList<String>();
                if (accept("|")) {
                    do {
                        var conditionalVariableName = identifier();
                        conditionalVariableNames.add(conditionalVariableName);
                    } while (accept(","));
                }
                expect(")");

                expect("{");
                var conditionalDistProbs = new ArrayList<List<Double>>();
                var conditionalDistValues = new ArrayList<List<String>>();
                if (accept("table")) {
                    var probs = new ArrayList<Double>();
                    do {
                        double num = floating();
                        probs.add(num);
                    } while (accept(","));
                    expect(";");

                    conditionalDistProbs.add(probs);
                    conditionalDistValues.add(new ArrayList<>());
                } else {
                    while (accept("(")) {
                        var values = new ArrayList<String>();
                        do {
                            var valueName = identifier();
                            values.add(valueName);
                        } while (accept(","));
                        expect(")");

                        var probs = new ArrayList<Double>();
                        do {
                            double num = floating();
                            probs.add(num);
                        } while (accept(","));
                        expect(";");

                        conditionalDistValues.add(values);
                        conditionalDistProbs.add(probs);
                    }
                }
                expect("}");

                tableVariableNames.add(variableName);
                tableConditionalVariableNames.add(conditionalVariableNames);
                tableConditionalProbs.add(conditionalDistProbs);
                tableConditionalValues.add(conditionalDistValues);
            } else {
                break;
            }
        }
        // :( :(
        // Take parsed values and turn them into a network.
        var tables = new ArrayList<Table>();
        var dependencyGraph = new ArrayList<List<Integer>>();
        for (int i = 0; i < variables.size(); i++) {
            dependencyGraph.add(new ArrayList<>());
        }
        for (int i = 0; i < variables.size(); i++) {
            var variableName = variableNames.get(i);
            int j = tableVariableNames.indexOf(variableName);

            var conditionalVariableNames = tableConditionalVariableNames.get(j);
            var dependencies = dependencyGraph.get(i);
            var tableVarIds = new ArrayList<Integer>();
            var shape = new ArrayList<Integer>();
            for (var conditionalVariableName : conditionalVariableNames) {
                int conditionalVariableId = variableNames.indexOf(conditionalVariableName);
                dependencies.add(conditionalVariableId);
                tableVarIds.add(conditionalVariableId);
                var conditionalVariable = variables.get(conditionalVariableId);
                shape.add(conditionalVariable.getValNames().size());
            }
            tableVarIds.add(i);
            shape.add(variables.get(i).getValNames().size());

            // Set probabilities into the NDArray

            var strides = new ArrayList<Integer>();
            int stride = 1;
            for (int axis = shape.size() - 1; axis >= 0; axis--) {
                strides.add(stride);
                stride *= shape.get(axis);
            }
            Collections.reverse(strides);
            int totalSize = stride;

            // disgusting
            var nums = new double[totalSize];
            var scenarios = tableConditionalValues.get(j);
            var condDist = tableConditionalProbs.get(j);
            for (int i1 = 0; i1 < scenarios.size(); i1++) {
                var scenario = scenarios.get(i1);
                int offset = 0;
                for (int axis = 0; axis < scenario.size(); axis++) {
                    var conditionalValueName = scenario.get(axis);
                    int conditionalVariableId = tableVarIds.get(axis);
                    var valueId = variables.get(conditionalVariableId).getValIds().get(conditionalValueName);
                    offset += strides.get(axis) * valueId;
                }
                var dist = condDist.get(i1);
                for (int k = 0; k < dist.size(); k++) {
                    double p = dist.get(k);
                    nums[offset + k] = p;
                }
            }

            tables.add(new Table(tableVarIds, new NDArray(shape, nums)));
        }
        return new BayesianNetwork(dependencyGraph, variables, variableNames, tables);
    }

    private void whitespace() {
        while (pos < buf.length && Character.isWhitespace(buf[pos])) {
            pos++;
        }
    }

    private void integer() {
        whitespace();
        while (pos < buf.length && Character.isDigit(buf[pos])) {
            pos++;
        }
    }

    private double floating() throws ParseException {
        whitespace();
        int end = pos;
        if (end >= buf.length || !Character.isDigit(buf[end]))
            throw new ParseException("Expected a decimal", end);
        end++;
        if (end >= buf.length || buf[end] != '.')
            throw new ParseException("Expected a decimal", end);
        end++;
        if (end >= buf.length || !Character.isDigit(buf[end]))
            throw new ParseException("Expected a decimal", end);
        while (end < buf.length && Character.isDigit(buf[end])) {
            end++;
        }
        double res = Double.parseDouble(String.copyValueOf(buf, pos, end - pos));
        pos = end;
        return res;
    }


    private void expect(String s) throws ParseException {
        whitespace();
        for (int i = 0; i < s.length(); i++) {
            if (pos + i < buf.length && s.charAt(i) != buf[pos + i]) {
                throw new ParseException("Expected '" + s + "'", pos + i);
            }
        }
        pos += s.length();
    }

    private boolean accept(String s) {
        whitespace();
        for (int i = 0; i < s.length(); i++) {
            if (pos + i >= buf.length || s.charAt(i) != buf[pos + i])
                return false;
        }
        pos += s.length();
        return true;
    }

    private String identifier() throws ParseException {
        whitespace();
        int end = pos;
        if (end >= buf.length || (!Character.isLetter(buf[end]) && buf[end] != '_'))
            throw new ParseException("Expected an identifier", end);
        end++;
        while (end < buf.length && (Character.isLetterOrDigit(buf[end]) || buf[end] == '_')) {
            end++;
        }
        var res = String.copyValueOf(buf, pos, end - pos);
        pos = end;
        return res;
    }
}

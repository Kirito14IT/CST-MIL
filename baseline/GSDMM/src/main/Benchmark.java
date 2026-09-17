package main;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.util.HashMap;
import java.util.Random;

/** SCCD adapter: original collapsed Gibbs model plus frozen-count inference. */
public class Benchmark {
    private static HashMap<String, Integer> vocabulary(int size) {
        HashMap<String, Integer> map = new HashMap<String, Integer>();
        for (int i = 0; i < size; i++) map.put(Integer.toString(i), i);
        return map;
    }

    public static void main(String[] args) throws Exception {
        if (args.length == 0) throw new IllegalArgumentException("Expected est or inf");
        if (args[0].equals("est")) {
            // est K V alpha beta iterations seed input.txt model.bin phi.txt
            if (args.length != 10) throw new IllegalArgumentException("est argument count");
            int k = Integer.parseInt(args[1]);
            int v = Integer.parseInt(args[2]);
            Model model = new Model(k, v, Integer.parseInt(args[5]),
                    Double.parseDouble(args[3]), Double.parseDouble(args[4]), "sccd", "");
            model.random = new Random(Long.parseLong(args[6]));
            DocumentSet docs = DocumentSet.fromPlain(args[7], vocabulary(v));
            model.intialize(docs);
            model.gibbsSampling(docs);
            try (ObjectOutputStream out = new ObjectOutputStream(new FileOutputStream(args[8]))) {
                out.writeObject(model);
            }
            try (BufferedWriter out = Files.newBufferedWriter(Paths.get(args[9]), StandardCharsets.UTF_8)) {
                for (int z = 0; z < k; z++) {
                    for (int w = 0; w < v; w++) {
                        if (w > 0) out.write(" ");
                        out.write(Double.toString((model.n_zv[z][w] + model.beta)
                                / (model.n_z[z] + model.beta0)));
                    }
                    out.newLine();
                }
            }
            System.out.println("trained documents=" + docs.D + " topics=" + k + " vocabulary=" + v);
        } else if (args[0].equals("inf")) {
            // inf model.bin input.txt output.txt
            if (args.length != 4) throw new IllegalArgumentException("inf argument count");
            Model model;
            try (ObjectInputStream in = new ObjectInputStream(new FileInputStream(args[1]))) {
                model = (Model) in.readObject();
            }
            DocumentSet docs = DocumentSet.fromPlain(args[2], vocabulary(model.V));
            try (BufferedWriter out = Files.newBufferedWriter(Paths.get(args[3]), StandardCharsets.UTF_8)) {
                for (Document doc : docs.documents) {
                    double[] probabilities = model.posterior(doc);
                    for (int z = 0; z < model.K; z++) {
                        if (z > 0) out.write(" ");
                        out.write(Double.toString(probabilities[z]));
                    }
                    out.newLine();
                }
            }
        } else {
            throw new IllegalArgumentException("Unknown mode " + args[0]);
        }
    }
}

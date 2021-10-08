import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.tribuo.classification.evaluation.LabelEvaluator;
import org.tribuo.common.sgd.AbstractLinearSGDTrainer;
import org.tribuo.common.sgd.AbstractSGDTrainer;
import org.tribuo.ensemble.BaggingTrainer;
import org.tribuo.evaluation.Evaluation;
import org.tribuo.regression.evaluation.RegressionEvaluator;
import org.tribuo.regression.sgd.linear.LinearSGDTrainer;
import org.tribuo.regression.slm.SLMTrainer;

import java.io.File;
import java.net.URL;
import java.util.HashMap;
import java.util.logging.Level;
import java.util.logging.Logger;

public class TestReproductionTest {

    @BeforeAll
    public static void setup() {
        Class<?>[] classes = new Class<?>[]{AbstractSGDTrainer.class, AbstractLinearSGDTrainer.class, LinearSGDTrainer.class, BaggingTrainer.class, SLMTrainer.class};
        for (Class<?> c : classes) {
            Logger logger = Logger.getLogger(c.getName());
            logger.setLevel(Level.WARNING);
        }
    }

    @Test
    public void testRegression() throws Exception {
        URL uModels = TestReproductionTest.class.getResource("/models/regression");
        URL uData = TestReproductionTest.class.getResource("/data/winequality-red.csv");
        File directoryPath = new File(uModels.getFile());
        System.out.println(directoryPath);
        String provFiles[] = directoryPath.list();

        for(int i=0; i<provFiles.length; i++) {
            HashMap<String, Evaluation> evals = TestReproduction.reproduceModel(provFiles[i],
                    uModels.getFile(),
                    uData.getFile(),
                    new RegressionEvaluator());

            Assertions.assertEquals(evals.get("old").asMap(), evals.get("new").asMap());
        }
    }

    @Test
    public void testClassification() throws Exception {
        URL uModels = TestReproductionTest.class.getResource("/models/classification");
        URL uData = TestReproductionTest.class.getResource("/data/bezdekIris.data");
        File directoryPath = new File(uModels.getFile());
        System.out.println(directoryPath);
        String provFiles[] = directoryPath.list();

        for(int i=0; i<provFiles.length; i++) {
            HashMap<String, Evaluation> evals = TestReproduction.reproduceModel(provFiles[i],
                    uModels.getFile(),
                    uData.getFile(),
                    new LabelEvaluator());

            Assertions.assertEquals(evals.get("old").asMap(), evals.get("new").asMap());
        }
    }
}

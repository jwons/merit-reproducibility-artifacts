import java.io.FileInputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;

import org.tribuo.*;
import org.tribuo.evaluation.Evaluation;
import org.tribuo.evaluation.Evaluator;
import org.tribuo.evaluation.TrainTestSplitter;
import org.tribuo.data.csv.CSVLoader;
import org.tribuo.datasource.ListDataSource;
import org.tribuo.evaluation.TrainTestSplitter;
import org.tribuo.classification.*;
import org.tribuo.classification.evaluation.*;
import org.tribuo.classification.sgd.linear.LogisticRegressionTrainer;
import org.tribuo.classification.sgd.linear.LinearSGDModel;
import org.tribuo.math.optimisers.*;
import org.tribuo.regression.*;
import org.tribuo.regression.evaluation.*;
import org.tribuo.regression.sgd.RegressionObjective;
import org.tribuo.regression.sgd.linear.LinearSGDTrainer;
import org.tribuo.regression.sgd.objectives.SquaredLoss;
import org.tribuo.regression.rtree.CARTRegressionTrainer;
import org.tribuo.regression.rtree.impurity.MeanSquaredError;
import org.tribuo.regression.xgboost.XGBoostRegressionTrainer;
import org.tribuo.util.Util;
import org.tribuo.provenance.DatasetProvenance;
import org.tribuo.provenance.ModelProvenance;
import org.tribuo.reproducibility.ReproUtil;
import org.tribuo.transform.*;
import org.tribuo.transform.transformations.LinearScalingTransformation;
import org.tribuo.classification.*;
import org.tribuo.classification.evaluation.*;
import com.oracle.labs.mlrg.olcut.config.Configurable;
import com.oracle.labs.mlrg.olcut.config.ConfigurationManager;
import com.oracle.labs.mlrg.olcut.config.DescribeConfigurable;
import com.oracle.labs.mlrg.olcut.provenance.*;
import com.oracle.labs.mlrg.olcut.provenance.primitives.*;
import com.oracle.labs.mlrg.olcut.config.json.JsonConfigFactory;
import com.oracle.labs.mlrg.olcut.config.property.SimpleProperty;

public class TestReproduction {
    public static HashMap<String, Evaluation> reproduceModel(String filename, String directoryPathName, String newFilePath, Evaluator evaluator) throws Exception{
        System.out.println(filename);
        Model oldModel = null;
        try {
            FileInputStream fileIn = new FileInputStream(directoryPathName + "/" + filename);
            ObjectInputStream in = new ObjectInputStream(fileIn);
            oldModel = (Model) in.readObject();
            in.close();
            fileIn.close();
        } catch (IOException e) {
            e.printStackTrace();
        } catch (ClassNotFoundException c) {
            System.out.println("Model class not found");
            c.printStackTrace();
        }

        ReproUtil repro = new ReproUtil(oldModel);
        ArrayList<String> componentNames = new ArrayList<String>(repro.getConfigurationManager().getComponentNames());
        String sourceKey = null;
        for (String name : componentNames){
            if(name.length() > 13 && "csvdatasource".equals(name.substring(0, 13))){
                sourceKey = name;
            }
        }
        repro.getConfigurationManager().overrideConfigurableProperty(sourceKey, "dataPath", new SimpleProperty(newFilePath));
        Model newModel = repro.reproduceFromProvenance();

        var splitter = new TrainTestSplitter((DataSource) repro.getConfigurationManager().lookup(sourceKey), 0.7f, 0L);

        var oldEvaluation = evaluator.evaluate(oldModel,splitter.getTest());
        var newEvaluation = evaluator.evaluate(newModel,splitter.getTest());
        HashMap<String, Evaluation> evaluations = new HashMap<>();
        evaluations.put("old", oldEvaluation);
        evaluations.put("new", newEvaluation);
        return (evaluations);
    }
}

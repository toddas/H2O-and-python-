import h2o
import subprocess
from IPython.display import Image



#funkcijos visualizacijai

#ini ir duom uzkrovimas
h2o.init()
df = h2o.import_file("creditcard.csv")



#pradiniai seedai ir medziai
seed = 12345
ntrees = 100
#
isoforest = h2o.estimators.H2OIsolationForestEstimator(ntrees=ntrees,
                                                       seed=seed)
isoforest.train(x=df.col_names[0:31],
                training_frame=df)
predictions = isoforest.predict(df)

h2o_predictions = predictions.as_data_frame()

quantile = 0.95
quantile_frame = predictions.quantile([quantile])


print(quantile_frame)

threshold = quantile_frame[0, "predictQuantiles"]
predictions["predicted_class"] = predictions["predict"] > threshold
predictions["class"] = df["Class"]
print(predictions)











import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.metrics import accuracy_score,confusion_matrix
from data_processing import *
from model import *

x_test = test_feat.drop('lang',axis=1)
y_test = test['lang']

#get pridiction on test set
labels=model.predict_classes(x_test)
predictions=encoder.inverse_transform(labels)

#accuracy on test set 
accuarcy = accuracy_score(y_test,predictions)
print(accuracy)


#create confusion matrix
conf_matrix=confusion_matrix(y_test,predictions)
confusion_matrix_df=pd.DataFrame(conf_matrix,columns=lang,index=lang)



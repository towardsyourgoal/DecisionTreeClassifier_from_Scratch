import numpy as np
from collections import Counter

class Node:
    """
    This class is the basic building block of the Decision tree. 
    Each node stores the information needed to make a split or predictions
    feature_index - Index of the feature to split on
    threshold - Threshold value for the split
    left - Left child node
    right - Right child node
    value - mode value for leaf nodes for Classification
    """
    def __init__(self,feature_index=None,threshold=None,left=None,right=None,value=None,label_counts=None):
        self.feature_index=feature_index
        self.threshold=threshold
        self.left=left
        self.right=right
        self.value=value
        self.label_counts=label_counts

    def is_leaf_node(self):
        return self.value is not None


class DecisionTreeClassifier:
    """
    This class is the core which applies the logic for building a decision tree
    1. Initialization of parameters
    2. We will define entropy and information gain and use them to find the best split 
    4. Building tree with the given X and y
    5. Fit the classifier to the labeled data
    6. Make predictions of a batch
    7. Evaluation metrics
    """
    def __init__(self,max_depth=float("Inf"),min_samples_split=2):
        self.max_depth=max_depth
        self.min_samples_split=min_samples_split
        self.root=None
        self.confusion_matrix=None
        self.TP=0
        self.FP=0
        self.TN=0
        self.FN=0
        self.classes=None


    def entropy(self,y):
        """
        This function will calculate the impurity if a node
        """
        labels, label_count=np.unique(y,return_counts=True)
        prob_y=label_count/len(y)
        entropy=-(np.sum([p * np.log2(p) for p in prob_y if p>0 ]))
        return entropy

    def information_gain(self,parent_y,left_y,right_y):
        """
        This function as the name defines calculate how much pure information (reduction in impurity) we are getting on a specific split
        We will use this for each threshold to find the best split
        """
        parent_entropy=self.entropy(parent_y)
        n,n_left,n_right=len(parent_y),len(left_y),len(right_y)
        weighted_child_entropy=(n_left * self.entropy(left_y) + n_right * self.entropy(right_y))/n
        information_gain=parent_entropy - weighted_child_entropy
        return information_gain


    def calculate_leaf_value(self,y):
        """
        This give us the mode of the provided labels
        """
        y=list(y)
        return max(y,key=y.count)
    

    def find_best_split(self,X,y,feat_types):
        """
        This function returns information of the best split - 
        1. Index of the column  
        2. Threshold value of that column
        3. Indices of the dataset of the left child
        4. Indices of the dataset of the right child
        """
        best_feature,best_threshold,l_indices,r_indices=None,None,None,None
        n_samples, n_features=X.shape
        best_gain=-1
        for ix,feat_type in zip(range(n_features),feat_types):
            feature_data=X[:,ix]
            if feat_type== "object" or feat_type== "str":
                #To handle string or object type of data
                thresholds=np.unique(feature_data)
                for thr in thresholds:
                    left_indices=np.where(feature_data==thr)[0]
                    right_indices=np.where(feature_data!=thr)[0]
                    if len(left_indices)==0 or len(right_indices)==0:
                        continue

                    y_left,y_right=y[left_indices],y[right_indices]
                    gain=self.information_gain(y,y_left,y_right)
                    if gain>best_gain:
                        best_gain=gain
                        best_feature=ix
                        best_threshold=thr
                        l_indices=left_indices
                        r_indices=right_indices
            else:
                #To handle numeric type of data
                thresholds=np.unique(feature_data)
                for thr in thresholds:
                    left_indices=np.where(feature_data<=thr)[0]
                    right_indices=np.where(feature_data>thr)[0]
                    if len(left_indices)==0 or len(right_indices)==0:
                        continue
                    y_left, y_right=y[left_indices],y[right_indices]
                    gain=self.information_gain(y,y_left,y_right)
                    if gain > best_gain:
                        best_gain=gain
                        best_feature=ix
                        best_threshold=thr
                        l_indices=left_indices
                        r_indices=right_indices
        return best_feature,best_threshold,l_indices,r_indices


    def build_tree(self,X,y,feat_types,depth=0):
        """
        This function performs the recursion and build the decision tree with best split at each node
        This returns a built tree, which we will be using to traverse during the prediction
        """
        y_hash=np.ravel(y)
        counts=Counter(y_hash)
        n_samples=len(y)
        #Stopping criteria logic
        if self.max_depth and depth>=self.max_depth:
            return Node(value=self.calculate_leaf_value(y), label_counts=counts)
        elif n_samples<self.min_samples_split:
            return Node(value=self.calculate_leaf_value(y), label_counts=counts)
        elif len(np.unique(y))==1:
            return Node(value=self.calculate_leaf_value(y), label_counts=counts)
        
        best_feature,best_threshold,l_indices,r_indices=self.find_best_split(X,y,feat_types)

        if best_feature is None:
            return Node(value=self.calculate_leaf_value(y), label_counts=counts)

        feature_index=best_feature
        threshold=best_threshold
        left_indices=l_indices
        right_indices=r_indices

        left_child=self.build_tree(X[left_indices],y[left_indices],feat_types,depth+1)
        right_child=self.build_tree(X[right_indices],y[right_indices],feat_types,depth+1)

        return Node(feature_index,threshold,left_child,right_child)


    def fit(self,X,y):
        """
        This functions fits the trained tree to the given data
        The trained  tree is assigned to the root node
        We will be interacting with this function on the front-end while using this algorithm
        """
        Xt=X.to_numpy()
        yt=y.to_numpy().reshape(-1,1)
        feature_types=X.dtypes
        self.classes=np.unique(yt)
        self.root=self.build_tree(Xt,yt,feature_types)


    def traverse_tree(self,x,node,col_types):
        """
        This function is Used to make prediction for a single sample at a time
        The test data traverse through the trained decision tree and then the prediction is made
        """
        feat_types=list(col_types)
        if node.is_leaf_node():
            return node.value
        feat_index=node.feature_index
        feature_data=x[feat_index]
        
        if feat_types[feat_index] == 'object' or feat_types[feat_index] == 'str' :
            if feature_data==node.threshold:
                return self.traverse_tree(x,node.left,col_types)
            else:
                return self.traverse_tree(x,node.right,col_types)
        else:
            if feature_data<=node.threshold:
                return self.traverse_tree(x,node.left,col_types)
            else:
                return self.traverse_tree(x,node.right,col_types)


    def predict(self,X):
        """
        This takes the batch of test or validation data to make predictions
        Returns the predictions
        """
        col_types=X.dtypes
        xtest=X.to_numpy()
        predictions=np.array([self.traverse_tree(x,self.root,col_types) for x in xtest])
        predictions=predictions.ravel()
        return predictions
    
    def traverse_tree_for_probs(self,x,node,col_types):
        feat_types=list(col_types)
        feat_index=node.feature_index
        feature_data=x[feat_index]
        if node.is_leaf_node():
            total_samples=sum(node.label_counts.values())
            probs= {label: count/total_samples for label, count in node.label_counts.items()}
            probs_l=[probs.get(cls,0.0) for cls in self.classes]
            return probs_l
        
        if feat_types[feat_index] == 'object' or feat_types[feat_index] == 'str' :
            if feature_data==node.threshold:
                return self.traverse_tree_for_probs(x,node.left,col_types)
            else:
                return self.traverse_tree_for_probs(x,node.right,col_types)
        else:
            if feature_data<=node.threshold:
                return self.traverse_tree_for_probs(x,node.left,col_types)
            else:
                return self.traverse_tree_for_probs(x,node.right,col_types)
        
    
    def prodict_probs(self,X):
        col_types=X.dtypes
        xtest=X.to_numpy()
        probabilities=np.array([self.traverse_tree_for_probs(x,self.root,col_types) for x in xtest])
        # probabilities=probabilities.ravel()
        return probabilities


    def get_confusion_matrix(self,y_true,y_pred):
        """
        This function outputs a matrix that is sized num_of_classes x num_of_classes
        The rows are true values and columns are predicted values 
        """
        classes=np.unique(np.concatenate((y_true,y_pred)))
        num_classes=len(classes)
        self.confusion_matrix=np.zeros((num_classes,num_classes),dtype=int)
        class_to_index={cls:i for i,cls in enumerate(classes)}
        for i,j in zip(y_true,y_pred):
            true_idx=class_to_index[i]
            pred_idx=class_to_index[j]
            self.confusion_matrix[true_idx,pred_idx]+=1
        return self.confusion_matrix
    

    def find_metrics_from_con_mat(self,y_true,y_pred,con_mat):
        """
        This function returns core metrics 
        TP- True Positoves,
        TN- True Negatives,
        FP- False Positives
        FN- False Negatives
        These will be used in finding important evaluation metrics for classification
        """
        classes=np.unique(np.concatenate((y_true,y_pred)))
        n_classes=len(classes)
        if con_mat.shape == (2,2):
            self.TP=con_mat[1,1]
            self.TN=con_mat[0,0]
            self.FP=con_mat[0,1]
            self.FN=con_mat[1,0]
        else:
            for c, class_label in enumerate(classes):
                tp=con_mat[c,c]
                fp=np.sum(con_mat[:,c]) - tp
                fn=np.sum(con_mat[c,:]) - tp
                tn=np.sum(con_mat) - tp - fp - fn 
                self.TP+=tp
                self.FP+=fp
                self.FN+=fn
                self.TN+=tn
        return self.TP, self.TN, self.FP, self.FN
    
    def accuracy_score(self,y_true,y_pred):
        """
        This returns accuracy of the classfier 
        How well the model is predicting correctly across the samples
        """
        n_samples=len(y_true)
        con_mat=self.get_confusion_matrix(y_true,y_pred)
        TP, TN, FP, FN=self.find_metrics_from_con_mat(y_true,y_pred,con_mat)
        if con_mat.shape==(2,2):
            return (TP+TN)/(TP+TN+FP+FN)
        else:
            return TP/n_samples
    
    def precision_score(self,y_true,y_pred):
        """
        This returns how often a positive label is actually correct
        Our aim is to make the FP to 0 - no false alarms 
        """
        con_mat=self.get_confusion_matrix(y_true,y_pred)
        TP, TN, FP, FN=self.find_metrics_from_con_mat(y_true,y_pred,con_mat)
        if con_mat.shape==(2,2):
            return TP/(TP+FP)
        else:
            return TP/(TP+FP)    


    def recall_score(self,y_true,y_pred):
        """
        This returns score - "of all the positive instances, how much the model actually identified correctly ?"
        Very important for imbalanced datasets where missing a positive can be very costly
        Our aim is to make the FN to 0 - no missed alarms 
        """
        con_mat=self.get_confusion_matrix(y_true,y_pred)
        TP, TN, FP, FN=self.find_metrics_from_con_mat(y_true,y_pred,con_mat)
        if con_mat.shape==(2,2):
            return TP/(TP+FN)
        else:
            return TP/(TP+FN)

    def f1_score(self,y_true,y_pred, average='micro'):
        """
        This calculates hormonice mean of precision and recall
        Again, very important for imbalanced datasets
        Accounts for both False alarms as well as Missed alarms
        """
        classes=np.unique(np.concatenate((y_true,y_pred)))
        n_classes=len(classes)
        macro_class_list=[]
        con_mat=self.get_confusion_matrix(y_true,y_pred)
        precision=self.precision_score(y_true,y_pred)
        recall=self.recall_score(y_true,y_pred)
        if con_mat.shape==(2,2):
            return 2 * (precision * recall)/ (precision + recall)
        else:
            if average=='micro':
                print("this gives micro averaged f1 score and for multiclass classification, it wiil be the same as accuracy, precision and recall")
                return precision
            elif average=='macro':
                for i in range(n_classes):
                    tp=con_mat[i,i]
                    fp=np.sum(con_mat[:,i]) - tp
                    fn=np.sum(con_mat[i,:]) - tp
                    tn=np.sum(con_mat) - tp - fp - fn
                    prec_i=tp/(tp+fp)
                    rec_i=tp/(tp+fn)
                    f1_score_i=2*prec_i*rec_i/(prec_i+rec_i)
                    macro_class_list.append(f1_score_i)
                return np.mean(macro_class_list)

    def fpr(self,y_true,y_pred):
        """
        This calculates and returns ratio of total False positives the model calculated to the actual negatives 
        Ideally should be zero, where there are no false alarms
        """
        classes=np.unique(np.concatenate((y_true,y_pred)))
        n_classes=len(classes)
        con_mat=self.get_confusion_matrix(y_true,y_pred)
        TP, TN, FP, FN=self.find_metrics_from_con_mat(y_true,y_pred,con_mat)
        if con_mat.shape==(2,2):
            return FP/(FP+TN)
        else:
            print("this is binary-only metric")

    def specificity(self,y_true,y_pred):
        """
        This is 1-fpr, basically True Negative Rate
        This denotes - how well a model is identifying actual negatives
        Ideally should be 1 (No false alarams)
        """
        classes=np.unique(np.concatenate((y_true,y_pred)))
        n_classes=len(classes)
        con_mat=self.get_confusion_matrix(y_true,y_pred)
        TP, TN, FP, FN=self.find_metrics_from_con_mat(y_true,y_pred,con_mat)
        if con_mat.shape==(2,2):
            return TN/(TN+FP)
        else:
            print("this is binary-only metric")


    def calculate_roc_auc_score(self,y_true,y_pred,pos_probs):
        yt=np.array(y_true)
        ytprobs=np.array(pos_probs)
        classes=np.unique(np.concatenate((y_true,y_pred)))
        n_classes=len(classes)
        assert n_classes==2
        test_pos_probabilities=pos_probs
        sorted_indices = np.argsort(test_pos_probabilities)[::-1]
        y_true_sorted = yt[sorted_indices]
        y_probs_sorted = ytprobs[sorted_indices]
        n_pos=(y_true==1).sum()
        n_neg=(y_true==0).sum()
        fpr_list = [0.0]
        tpr_list = [0.0]
        for i in range(len(y_probs_sorted)):
            TP=(y_true_sorted[:i+1]==1).sum()
            FP=(y_true_sorted[:i+1]==0).sum()
            tpr=TP/n_pos if n_pos>0 else 0
            fpr=FP/n_neg if n_neg>0 else 0
            tpr_list.append(tpr)
            fpr_list.append(fpr)
        tpr_list.append(1.0)
        fpr_list.append(1.0)
        roc_auc = np.trapezoid(tpr_list, fpr_list)
        return roc_auc

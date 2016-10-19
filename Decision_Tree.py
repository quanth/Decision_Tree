import numpy
from collections import Counter
import random

def read_datafile(fname, attribute_data_type='integer'):
    inf = open(fname, 'r')
    lines = inf.readlines()
    inf.close()
    # --
    X = []
    Y = []
    for l in lines:
        ss = l.strip().split(',')
        temp = []
        for s in ss:
            if attribute_data_type == 'integer':
                temp.append(int(s))
            elif attribute_data_type == 'string':
                temp.append(s)
            else:
                print("Unknown data type");
                exit();
        X.append(temp[:-1])
        Y.append(int(temp[-1]))
    return X, Y


class Node:
    #Tree node class, used to store the tree model.
    def __init__(self,index,value,attribute,parent,remaining_att,prediction):
        self.value = value                  #The spliting value at this node
        self.attribute = attribute          #The attribute used in splitting at this node (index)
        self.remaining_att = remaining_att  #The remaining attributes at this node
        #self.parent = parent               #The parent node, not needed
        #self.parent_prediction = None      #The prediction of Parent, not needed
        self.children = []                  #children nodes
        self.prediction = prediction        #Leave nodes have associated class prediction.
        self.index = index                  #The node's index in the list
        self.depth = 0                      #The node's depth
# ===
class DecisionTree:
    def __init__(self, split_random, depth_limit, curr_depth=0, default_label=1):
        self.split_random = split_random  # if True splits randomly, otherwise splits based on information gain
        self.depth_limit = depth_limit                                  #depth limit
        self.currentIndex = 0                                           #Index of the node in consideration
        attributes = range(len(X_train[0]))
        rootNode = Node(0,None,None,None,attributes,None)               #root node initialization
        self.NodeList = []                                              #node list
        self.NodeList.append(rootNode)


        print "Number of attributes: " + str(len(attributes))
        print "Depth Limit: " + str(self.depth_limit)
        print "Number of training sample: " + str(len(X_train))

    #==============================================================================
    #Entropy calculation
    #==============================================================================
    def entropy(self,Y_list):
        counter = Counter(Y_list)
        count_0 = counter[0]+0.0
        count_1 = counter[1]+0.0

        if count_0 == 0.0 or count_1 == 0.0:
            return 0

        prop_1 = count_1/(len(Y_list))
        prop_0 = count_0/(len(Y_list))

        return -prop_0*numpy.log2(prop_0) + -prop_1*numpy.log2(prop_1)

    #==============================================================================
    #Information gain calculation
    #==============================================================================
    def infoGain(self,X_list, Y_list, startingentropy, attribute_index):
        gain = startingentropy
        values = [0,1]
        tmp_dict = self.split_data_by_att(X_list,Y_list,attribute_index)
        for value in values:
            subset = tmp_dict[value]
            entropy_sub = self.entropy(subset[1])
            gain -= (len(subset[1])+0.0) / (len(Y_list)) * entropy_sub
        return gain

    #==============================================================================
    #split the data by attribute for training
    #==============================================================================
    def split_data_by_att(self, X_list, Y_list, attribute_index):
        values = [0,1]
        dict = {}
        for value in values:
            dict[value] = [[],[]]

        for i in range(len(X_list)):
            val = X_list[i][attribute_index]
            tag = Y_list[i]
            if val == 0:
                dict[0][0].append(X_list[i])
                dict[0][1].append(tag)
            else:
                dict[1][0].append(X_list[i])
                dict[1][1].append(tag)
        return dict

    #==============================================================================
    #Majority class
    #==============================================================================
    def majority(self,counter):
        if counter[0] > counter[1]:
            return 0
        return 1

    #==============================================================================
    #Train function
    #==============================================================================
    def train(self, X_train, Y_train):
        # receives a list of objects of type Example
        # TODO: implement decision tree training
        assert len(X_train) == len(Y_train)
        #print "============================="

        counter = Counter(Y_train)
        currentNode = self.NodeList[self.currentIndex]

        #terminal cases
        # ==============================================================================
        if counter[0] == 0 or counter[0] == len(Y_train) or currentNode.depth > self.depth_limit or len(currentNode.remaining_att) < 1 :
            currentNode.prediction = self.majority(counter)
            return

        #choose attribute to split
        # ==============================================================================
        bestVal = -float("inf")
        bestInd = currentNode.remaining_att[0]
        startingEntropy = self.entropy(Y_train)

        if(self.split_random):
            bestInd = random.choice(currentNode.remaining_att)
        else:
            for att_index in currentNode.remaining_att:
                val = self.infoGain(X_train,Y_train,startingEntropy,att_index)
                if val > bestVal:
                    bestInd = att_index
                    bestVal = val

        currentNode.attribute = bestInd


        # split data, remove used attribute
        # ==============================================================================
        dictionary = self.split_data_by_att(X_train,Y_train,bestInd)
        remaining_att = currentNode.remaining_att[:]
        remaining_att.remove(bestInd)


        # add new nodes to the tree
        # ==============================================================================
        Node_0 = Node(len(self.NodeList)+0, 0, None, self.currentIndex, remaining_att[:], None)
        Node_1 = Node(len(self.NodeList)+1, 1, None, self.currentIndex, remaining_att[:], None)

        currentNode.children = [len(self.NodeList)+0,len(self.NodeList)+1]
        self.NodeList.extend([Node_0,Node_1])

        # recursion call
        # ==============================================================================
        self.currentIndex = Node_0.index
        Node_0.depth = currentNode.depth+1
        self.train(dictionary[0][0],dictionary[0][1])

        self.currentIndex = Node_1.index
        Node_1.depth = currentNode.depth+1
        self.train(dictionary[1][0], dictionary[1][1])

        return


    def predict(self, x):
    # receives a list of booleans
    # TODO: implement decision tree prediction
    # ===
        #for node in self.NodeList:
            #print node.index
            #print node.children
            #print node.parent_prediction
            #print node.prediction
            #x = 0
            #if node.prediction != None:
                #print node.index
                #print node.prediction
        #Y_predict = []
        '''for i in range(len(X_train)):
            self.currentIndex = 0
            instance_att = X_train[i]
            prediction = self.NodeList[self.currentIndex].prediction
            Node_used = []
            while prediction is None:
                Node_used.append(self.currentIndex)
                val = instance_att[self.NodeList[self.currentIndex].attribute]
                self.currentIndex = self.NodeList[self.currentIndex].children[val]
                prediction = self.NodeList[self.currentIndex].prediction
            print "============"
            print Node_used
            Y_predict.append(prediction)'''
        self.currentIndex = 0
        instance_att = x
        prediction = self.NodeList[self.currentIndex].prediction
        Node_used = []

        # continue searching until a prediction is found (at leave nodes)
        # ==============================================================================
        while prediction is None:
            Node_used.append(self.currentIndex)
            val = instance_att[self.NodeList[self.currentIndex].attribute]
            self.currentIndex = self.NodeList[self.currentIndex].children[val]
            prediction = self.NodeList[self.currentIndex].prediction

        return prediction


def compute_accuracy(dt_classifier, X_test, Y_test):
    numRight = 0
    for i in range(len(Y_test)):
        x = X_test[i]
        y = Y_test[i]
        if y == dt_classifier.predict(x):
            numRight += 1
    return (numRight * 1.0) / len(Y_test)


# ==============================================
# ==============================================
# TODO: write your code

# test accuracy
# ==============================================================================
print "=========================================="
print "WITH IG SPLIT"
print "=========================================="
X_train, Y_train = read_datafile('train.txt')
X_test, Y_test = read_datafile('test.txt')
dt = DecisionTree(split_random=False,depth_limit=1000)
dt.train(X_train,Y_train)
print "Test data acc:"+str(compute_accuracy(dt,X_test,Y_test))
X_test = X_train
Y_test = Y_train
print "Train data acc:"+str(compute_accuracy(dt,X_test,Y_test))
print "=========================================="
print "RANDOM SPLIT"
print "=========================================="
X_train, Y_train = read_datafile('train.txt')
X_test, Y_test = read_datafile('test.txt')
dt = DecisionTree(split_random=True,depth_limit=1000)
dt.train(X_train,Y_train)
print "Test data acc:"+str(compute_accuracy(dt,X_test,Y_test))
X_test = X_train
Y_test = Y_train
print "Train data acc:"+str(compute_accuracy(dt,X_test,Y_test))
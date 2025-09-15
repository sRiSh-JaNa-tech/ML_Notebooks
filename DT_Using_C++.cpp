#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <map>
#include <unordered_map>
#include <Eigen/Dense>
#include <set>

template <typename T>
struct TreeNode{
    T threshold = 0; //containes the threshold used for the split 
    double gini = 0; //impurity in the current split 
    int samples = 0; //size of the data in present in the current node 
    int featureind = -1; //best feature to split 
    std::string node = ""; //for display purpose to know on which node am i 
    std::vector<std::string> dataLabels = {}; //main data split for the node 
    std::vector<int> left; //Unique indexes for the left tree value<=threshold 
    std::vector<int> right; //Unique indexes for the right tree !(value<=threshold)
    std::vector<std::string> leftLabels; //this will contain all the labels of the left indexs
    std::vector<std::string> rightLabels; //this will contain all the labels of the right indexes 
    TreeNode<T>* toleft; //pointer to the left 
    TreeNode<T>* toright; //pointer to the right 
    TreeNode(){ // Node initialization 
        toleft = toright = nullptr;
    }
};
double giniImpurity(const std::vector<std::string>& labels){
    std::unordered_map<std::string, int> freq;
    for(const auto& name : labels){
        freq[name]++;
    }
    double impurity = 1.0;
    int total = labels.size();
    for(const auto& pair : freq){
        double p = (double)pair.second / total;
        impurity -= p;
    }
    return impurity;
}
template <typename Y>
TreeNode<Y>* build_right_Tree(TreeNode<Y>* parent,Y threshold){
    TreeNode<Y>* root = new TreeNode<Y>();
    root->threshold = threshold;
    root->gini = giniImpurity(parent->rightLabels);
    root->samples = parent->rightLabels.size();
    root->node = "right";
    root->dataLabels = parent->rightLabels;
    return root;
}
template <typename U>
TreeNode<U>* build_left_Tree(TreeNode<U>* parent,U threshold){
    TreeNode<U>* root = new TreeNode<U>();
    root->threshold = threshold;
    root->gini = giniImpurity(parent->leftLabels);
    root->samples = parent->leftLabels.size();
    root->node = "left";
    root->dataLabels = parent->leftLabels;
    return root;
}
template <typename U>
auto s_n_p(const std::vector<std::vector<std::string>>& data,TreeNode<U>* parent, int featureind,U threshold,int multi) -> double{
    for(auto i = 0;i < static_cast<int>(data.size());++i){
        float val = stof(data[i][featureind]);
        if(val <= threshold){
            parent->left.push_back(i);
        }else{
            parent->right.push_back(i);
        }
    }
    std::vector<std::string> parentLabels;
    for(const auto& row : data){
        parentLabels.push_back(row.back());
    }
    int impurity = giniImpurity(parentLabels);
    ;
    for(const auto idx : parent->left){
        parent->leftLabels.push_back(data[idx].back());
    }
    for(const auto idx : parent->right){
        parent->rightLabels.push_back(data[idx].back());
    }
    double Limpurity = giniImpurity(parent->leftLabels);
    double Rimpurity = giniImpurity(parent->rightLabels);
    double weightedGini = ((parent->leftLabels.size())*Limpurity + (parent->rightLabels.size())*Rimpurity) / static_cast<int>(data.size());
    double infogain = impurity - weightedGini;
    if(multi == 0){
        return infogain;
    }
    else if(multi == 1){
        return 0;
    }
    return 0;
}
std::vector<float> SortingFeature(const std::vector<std::vector<std::string>>& data,int featureIndex){
    std::set<float> uniqueVals;
    for(const auto& row : data){
        uniqueVals.insert(std::stof(row[featureIndex]));
    }
    return std::vector<float>(uniqueVals.begin(),uniqueVals.end());
}
std::vector<float> getThreshold(const std::vector<float>& sortedVals){
    std::vector<float> thresholds;
    for(size_t i = 0;i < sortedVals.size() - 1;++i){
        float mid = (sortedVals[i] + sortedVals[i+1]) / 2.0;
        thresholds.push_back(mid);
    }
    return thresholds;
}
template <typename I>
TreeNode<I>* split_tree(TreeNode<I>* parent, const std::vector<std::vector<std::string>>& data) {
    int featureCount = data[0].size() - 1; // last column is label
    std::map<double, std::pair<int, I>> infogains; // (feature, threshold)
    for (int i = 0; i < featureCount; ++i) {
        std::vector<float> sortedvals = SortingFeature(data, i);
        std::vector<float> thresholds = getThreshold(sortedvals);
        for (const auto& thres : thresholds) {
            parent->left.clear();
            parent->right.clear();
            parent->leftLabels.clear();
            parent->rightLabels.clear();
            double gain = s_n_p(data, parent, i, thres, 0);
            infogains[gain] = {i, thres}; //{gain : (featureIndex, threshold)}
        }
    }
    if (infogains.empty()) return parent; // No split found here so best infogain 
    //will terminate the program when the there is no more to split 
    //good for recursive approach 
    auto best = infogains.rbegin(); // bst infogain
    int bestFeatureIndex = best->second.first;
    I bestThreshold = best->second.second;
    parent->left.clear();
    parent->right.clear();
    parent->leftLabels.clear();
    parent->rightLabels.clear();
    s_n_p(data, parent, bestFeatureIndex, bestThreshold, 1); //calling it to fill the left and right data for right and left trees 
    parent->featureind = bestFeatureIndex; // building the left and right nodes for the parent 
    parent->threshold = bestThreshold;
    parent->toleft = build_left_Tree(parent, bestThreshold);
    parent->toright = build_right_Tree(parent, bestThreshold);
    return parent;
}
template <typename M>
TreeNode<M>* Recursive_tree_builder(std::vector<std::vector<std::string>>& data, TreeNode<M>* root) {
    // Base case: if pure or too small to split
    if (giniImpurity(root->dataLabels) == 0.0 || root->dataLabels.size() <= 1) {
        return root;
    }
    // Perform the split
    split_tree(root, data);
    // Prepare child data
    std::vector<std::vector<std::string>> leftData, rightData;
    for (int idx : root->left) {
        leftData.push_back(data[idx]);
    }
    for (int idx : root->right) {
        rightData.push_back(data[idx]);
    }
    if (root->toleft) {
        root->toleft->dataLabels.clear();
        for (const auto& row : leftData) {
            root->toleft->dataLabels.push_back(row.back());
        }
        Recursive_tree_builder(leftData, root->toleft);
    }
    if (root->toright) {
        root->toright->dataLabels.clear();
        for (const auto& row : rightData) {
            root->toright->dataLabels.push_back(row.back());
        }
        Recursive_tree_builder(rightData, root->toright);
    }
    return root;
}

std::vector<double> MinMaxScaling(std::vector<std::vector<std::string>>& data,int featureindex){
    std::vector<double> Normalized;
    double max = std::stof(data[0][featureindex]);
    double min = std::stof(data[0][featureindex]);
    for(const auto& row : data){
        double num = std::stof(row[featureindex]);
        max = std::max(max,num);
        min = std::min(min, num);
    }
    int X;
    double formula; 
    for(const auto& row : data){
        X = std::stoi(row[featureindex]);
        formula = (X - min) / (max - min);
        Normalized.push_back(formula);
    }
    return Normalized;
}
std::vector<double> StandardScaling(std::vector<std::vector<std::string>>& data,int featureindex){
    std::vector<double> Normalised;
    std::vector<double> nums;
    int r_size = data.size();
    Eigen::VectorXd v(r_size);
    for (int i = 0; i < r_size; ++i) {
        v(i) = std::stof(data[i][featureindex]);
    }
    double mean = v.mean();
    double variance = (v.array() - mean).square().sum() / (v.size() - 1); 
    double stddev = std::sqrt(variance);
    int X,formula;
    for(const auto& row : data){
        X = std::stoi(row[featureindex]);
        formula = (X - mean) / stddev;
        Normalised.push_back(formula);
    }
    return Normalised;
}
std::vector<int> LabelEncoder(std::vector<std::vector<std::string>>& data,int featureindex){
    std::set<std::string> uniqueLabels;
    for(const auto& row : data){
        uniqueLabels.insert(row[featureindex]);
    }
    std::map<std::string, int> L_I;
    int index = 0;
    for(const auto& label : uniqueLabels){
        L_I[label] = index++;
    }
    std::vector<int> encoded;
    for(const auto& row : data){
        encoded.push_back(L_I[row[featureindex]]);
    }
    return encoded;
}
std::vector<std::vector<int>> OneHotEncoding(std::vector<std::vector<std::string>>& data,int featureindex){
    std::vector<std::vector<int>> encoded;
    std::set<std::string> uniqueLabels;
    for(const auto& row : data){
        uniqueLabels.insert(row[featureindex]);
    }
    std::map<std::string, int> labelToIndex;
    int index = 0;
    for (const auto& label : uniqueLabels) {
        labelToIndex[label] = index++;
    }
    int numLabels = uniqueLabels.size();
    for (const auto& row : data) {
        std::vector<int> oneHot(numLabels, 0);
        std::string value = row[featureindex];
        int pos = labelToIndex[value];
        oneHot[pos] = 1;
        encoded.push_back(oneHot);
    }
    return encoded;
}
template <typename L>
std::vector<std::string> predict(const std::vector<std::vector<std::string>>& data,TreeNode<L>* parent){
    std::vector<std::string> predict;
    for(const auto& row : data){
        TreeNode<L>* temp = parent;
        while(temp->toleft != nullptr && temp->toright != nullptr){
            L value = static_cast<L>(std::stof(row[temp->featureind]));
            if(value <= temp->threshold){
                temp = temp->toleft;
            }
            else{
                temp = temp->toright;
            }
        }
        std::unordered_map<std::string, int> freq;
        for (const auto& label : temp->dataLabels) {
            freq[label]++;
        }
        std::string prediction;
        int max_count = -1;
        for (const auto& pair : freq) {
            if (pair.second > max_count) {
                max_count = pair.second;
                prediction = pair.first;
            }
        }
        predict.push_back(prediction);
    }
    return predict;
}
void train_test_split(const std::vector<std::vector<std::string>>& data,std::vector<std::vector<std::string>>& X,std::vector<std::vector<std::string>>& Y,float test_ratio) {
    int total_size = data.size();
    int test_count = static_cast<int>(test_ratio * total_size);
    int train_count = total_size - test_count;

    for (int i = 0; i < train_count; ++i) {
        X.push_back(data[i]);
    }
    for (int i = train_count; i < total_size; ++i) {
        Y.push_back(data[i]);
    }
}
int main(){
    std::ifstream file("C:/Users/srish/Downloads/iris.csv");
    std::string line;
    std::vector<std::vector<std::string>> data;
    while(std::getline(file, line)){
        std::vector<std::string> row;
        std::stringstream ss(line);
        std::string value;
        while(std::getline(ss,value,',')){
            row.push_back(value);
        }
        data.push_back(row);
    } 
    std::vector<std::vector<std::string>> X,Y;
    float test_size = 0.2;
    train_test_split(data,X,Y,test_size);
    TreeNode<float>* parent = new TreeNode<float>();
    for(const auto& row : X){
        parent->dataLabels.push_back(row.back());
    }
    parent->samples = X.size();
    parent->node = "main";
    Recursive_tree_builder(X,parent);
    auto predictions = predict<float>(Y, parent);
    for(const auto& labels : predictions){
        std::cout<< labels <<std::endl;
    }
    return 0;
}
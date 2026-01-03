#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cctype>
#include <variant>
#include <algorithm>
#include <stdexcept>
#include <unordered_map>
#include <cfloat>
#include <random>
#include <cmath>
#include <numeric>
#include <iomanip>
#include <chrono>
using namespace std;
using namespace std::chrono;
using Cell = variant<int, double, string>;

class FileStream{
    private: 
        string link;
        vector<string> header;
        vector<vector<Cell>> dataframe;
        Cell parseValue(const string& value);
    public: 
    FileStream(string link){
        this->link = link;
    }
    FileStream(vector<vector<Cell>> df){
        this->dataframe = df;
    }
    void getFile(){
        ifstream file(this->link);
        if(!file.is_open()){
            throw runtime_error("Failed to open file");
        }
        string line,value;
        // Column Header 
        if(getline(file,line)){
            stringstream ss(line);
            while(getline(ss, value,',')){
                header.push_back(value);
            }
        }
        // Read Data
        while (getline(file, line)) {
            vector<Cell> row;
            stringstream ss(line);
            while (getline(ss, value, ',')) {
                row.push_back(parseValue(value));
            }
            dataframe.push_back(row);
        }
        file.close();
    }

    string toLower(string s) {
        transform(s.begin(), s.end(), s.begin(),
                [](unsigned char c) { return tolower(c); });
        return s;
    }

    void drop(vector<string> columns){
        for (const string& col : columns) {
            auto it = find(header.begin(), header.end(), col);
            if (it == header.end()) {
                throw runtime_error("Column not found: " + col);
            }
            int idx = it - header.begin();
            header.erase(it); // removes the header 
            for (auto &row : dataframe) { // erases the columns data
                if (idx >= static_cast<int>(row.size())) {
                    throw runtime_error("Row size mismatch while dropping: " + col);
                }
                row.erase(row.begin() + idx); 
            }
            cout << "Dropped column: " << col << endl;
        }
    }
    void drop(vector<string> columns, vector<vector<Cell>>& dataframe, vector<string>& header){
        for (const string& col : columns) {
            auto it = find(header.begin(), header.end(), col);
            if (it == header.end()) {
                throw runtime_error("Column not found: " + col);
            }
            int idx = it - header.begin();
            header.erase(it); // removes the header 
            for (auto &row : dataframe) { // erases the columns data
                if (idx >= static_cast<int>(row.size())) {
                    throw runtime_error("Row size mismatch while dropping: " + col);
                }
                row.erase(row.begin() + idx); 
            }
        }
    }

    vector<vector<Cell>> exclude(const vector<string>& columns) {
        vector<vector<Cell>> df_copy = dataframe;
        vector<string> h_copy = header;
        drop(columns, df_copy, h_copy);
        return df_copy;
    }

    vector<vector<Cell>> include(const vector<string>& columns) {
        vector<vector<Cell>> result;
        // Find column indices
        vector<int> indices;
        for (const string& col : columns) {
            auto it = find(header.begin(), header.end(), col);
            if (it == header.end())
                throw runtime_error("Column not found: " + col);
            indices.push_back(it - header.begin());
        }
        // Extract data
        for (const auto& row : dataframe) {
            vector<Cell> new_row;
            for (int idx : indices) {
                if (idx >= row.size())
                    throw runtime_error("Row size mismatch");
                new_row.push_back(row[idx]);
            }
            result.push_back(new_row);
        }
        return result;
    }

    void printHeader() {
        for (auto &h : this->header)
            cout << h << " | ";
        cout << endl;
    }

    void printData() {
        for (auto &r : this->dataframe) {
            for (auto& cell : r){
                visit([](auto&& val) {
                    cout << val << " ";
                }, cell);
            }
            cout << endl;
        }
    }
};

Cell FileStream::parseValue(const string& value) {
    try {
        size_t pos;
        int i = stoi(value, &pos);
        if (pos == value.size())
            return i;
    } catch (...) {}

    try {
        size_t pos;
        double d = stod(value, &pos);
        if (pos == value.size())
            return d;
    } catch (...) {}

    return value; // fallback to string
}

class PreProcessor{
    public:
        vector<vector<double>> toNumericMatrix(const vector<vector<Cell>>& df) {
            vector<vector<double>> numeric;
            for (size_t i = 0; i < df.size(); i++) {
                vector<double> row;
                for (size_t j = 0; j < df[i].size(); j++) {
                    const Cell& cell = df[i][j];

                    if (holds_alternative<int>(cell)) {
                        row.push_back(static_cast<double>(get<int>(cell)));
                    }
                    else if (holds_alternative<double>(cell)) {
                        row.push_back(get<double>(cell));
                    }
                    else {
                        throw runtime_error(
                            "Non-numeric value found at row " +
                            to_string(i) + ", column " + to_string(j)
                        );
                    }
                }
                numeric.push_back(row);
            }
            return numeric;
        }
        vector<int> labelEncode(const vector<vector<Cell>>& y_df, unordered_map<string, int>& label_map) {
            vector<int> y;
            int current_label = 0;

            for (size_t i = 0; i < y_df.size(); i++) {
                if (!holds_alternative<string>(y_df[i][0])) {
                    throw runtime_error("Label column must be string");
                }

                string label = get<string>(y_df[i][0]);

                // Assign new integer if label not seen before
                if (label_map.find(label) == label_map.end()) {
                    label_map[label] = current_label++;
                }

                y.push_back(label_map[label]);
            }
            return y;
        }
        vector<vector<double>> oneHotEncode(
            const vector<int>& y,
            int num_classes
        ) {
            vector<vector<double>> y_onehot(
                y.size(),
                vector<double>(num_classes, 0.0)
            );

            for (size_t i = 0; i < y.size(); i++) {
                int label = y[i];
                if (label < 0 || label >= num_classes) {
                    throw runtime_error("Invalid class index in one-hot encoding");
                }
                y_onehot[i][label] = 1.0;
            }

            return y_onehot;
        }
        vector<vector<double>> MinMaxScaler(const vector<vector<double>>& X) {
            if (X.empty())
                throw runtime_error("Empty matrix");

            int rows = X.size();
            int cols = X[0].size();

            vector<double> min_val(cols, DBL_MAX);
            vector<double> max_val(cols, -DBL_MAX);

            // ---------- Find min & max for each column ----------
            for (int i = 0; i < rows; i++) {
                if (static_cast<int>(X[i].size()) != cols){
                    throw runtime_error("Inconsistent row size");
                }
                for (int j = 0; j < cols; j++) {
                    min_val[j] = min(min_val[j], X[i][j]);
                    max_val[j] = max(max_val[j], X[i][j]);
                }
            }

            // ---------- Scale ----------
            vector<vector<double>> res(rows, vector<double>(cols));
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    if (max_val[j] == min_val[j]){
                        res[i][j] = 0.0;   // constant column
                    }
                    else{
                        res[i][j] = (X[i][j] - min_val[j]) / (max_val[j] - min_val[j]);
                    }
                }
            }
            return res;
        }
        void get_shape(vector<vector<double>>& X){
            cout << "(" << X.size() << ", " << X[0].size() << ")\n";
        }
        vector<vector<double>> Transpose(vector<vector<double>>& df){
            vector<vector<double>> res(df[0].size(), vector<double>(df.size()));
            for(size_t i = 0;i < df.size();i++){
                for(size_t j = 0;j < df[0].size();j++){
                    res[j][i] = df[i][j];
                }
            }
            return res;
        }
};

class Mat_Ops {
private:
    vector<vector<double>> df;
public:
    Mat_Ops(const vector<vector<double>>& entry) : df(entry) {}
    Mat_Ops(int r, int c) : df(r, vector<double>(c, 0.0)) {}

    int rows() const { return df.size(); }
    int cols() const { return df[0].size();}

    vector<vector<double>>& data() { return df; }
    const vector<vector<double>>& data() const { return df; }

    Mat_Ops dot(const Mat_Ops& B) const {
        if (cols() != B.rows())
            throw runtime_error("Shape mismatch in dot");

        Mat_Ops result(rows(), B.cols());

        for (int i = 0; i < rows(); i++)
            for (int j = 0; j < B.cols(); j++)
                for (int k = 0; k < cols(); k++)
                    result.df[i][j] += df[i][k] * B.df[k][j];

        return result;
    }

    Mat_Ops& add(const vector<vector<double>>& b) {
        //shape: (rows × 1)
        if (b.size() != static_cast<size_t>(rows()) || b[0].size() != 1) {
            throw runtime_error("Bias shape must be (rows x 1)");
        }

        for (int i = 0; i < rows(); i++) {
            for (int j = 0; j < cols(); j++) {
                df[i][j] += b[i][0];
            }
        }
        return *this;   // chaining
    }

    Mat_Ops T() const {
        Mat_Ops res(cols(), rows());
        for(int i=0;i<rows();i++)
            for(int j=0;j<cols();j++)
                res.data()[j][i] = df[i][j];
        return res;
    }

    void relu() {
        for (auto& row : df){
            for (double& v : row){
                if (v < 0) v = 0;
            }
        }
    }

    Mat_Ops stable_exp() const{
        Mat_Ops result(rows(),cols());
        for(int j = 0;j < cols();j++){
            double max_val = -DBL_MAX;
            for(int i = 0;i < rows();i++){
                max_val = max(max_val, df[i][j]);
            }
            for (int i = 0; i < rows(); i++) {
                result.data()[i][j] = std::exp(df[i][j] - max_val);
            }
        }
        return result;
    }
    Mat_Ops softmax_from_exp() const {
        Mat_Ops result(rows(), cols());
        for (int j = 0; j < cols(); j++) {
            double col_sum = 0.0;
            for (int i = 0; i < rows(); i++) {
                col_sum += df[i][j];
            }
            if (col_sum == 0.0)
                throw runtime_error("Softmax division by zero");
            for (int i = 0; i < rows(); i++) {
                result.data()[i][j] = df[i][j] / col_sum;
            }
        }
        return result;
    }

    vector<int> argmax_axis0() const {
        vector<int> preds(cols());
        // For each column (sample)
        for (int j = 0; j < cols(); j++) {
            double max_val = -DBL_MAX;
            int max_idx = 0;

            for (int i = 0; i < rows(); i++) {
                if (df[i][j] > max_val) {
                    max_val = df[i][j];
                    max_idx = i;
                }
            }
            preds[j] = max_idx;
        }
        return preds;
    }

    Mat_Ops sum_axis1() const {
        Mat_Ops result(rows(), 1);

        for (int i = 0; i < rows(); i++) {
            for (int j = 0; j < cols(); j++) {
                result.data()[i][0] += df[i][j];
            }
        }
        return result;
    }


    Mat_Ops clip(double eps) const {
        Mat_Ops result(rows(), cols());

        for (int i = 0; i < rows(); i++) {
            for (int j = 0; j < cols(); j++) {
                double v = df[i][j];
                if (v < eps) v = eps;
                if (v > 1.0 - eps) v = 1.0 - eps;
                result.data()[i][j] = v;
            }
        }
        return result;
    }

    double cross_entropy_loss(const Mat_Ops& y_true,
                          const Mat_Ops& y_pred) {
        if (y_true.rows() != y_pred.rows() ||
            y_true.cols() != y_pred.cols())
            throw runtime_error("Loss shape mismatch");

        double loss = 0.0;
        int batch_size = y_true.cols();

        for (int j = 0; j < batch_size; j++) {
            for (int i = 0; i < y_true.rows(); i++) {
                loss += y_true.data()[i][j] *
                        std::log(y_pred.data()[i][j]);
            }
        }

        return -loss / batch_size;
    }

    Mat_Ops operator - (const Mat_Ops& B) const {
        if (rows() != B.rows() || cols() != B.cols())
            throw runtime_error("Shape mismatch in subtraction");

        Mat_Ops result(rows(), cols());
        for (int i = 0; i < rows(); i++)
            for (int j = 0; j < cols(); j++)
                result.data()[i][j] = df[i][j] - B.data()[i][j];

        return result;
    }

    Mat_Ops operator / (double batch_size) const {
        Mat_Ops result(rows(),cols());
        for(int i = 0;i < rows();i++){
            for(int j = 0;j < cols();j++){
                result.data()[i][j] = df[i][j] / batch_size;
            }
        }
        return result;
    }
};

class Multi_Perceptron{
    private:
        vector<vector<double>> W1;
        vector<vector<double>> B1;
        vector<vector<double>> W2;
        vector<vector<double>> B2;
        vector<vector<double>> W3;
        vector<vector<double>> B3;
        vector<vector<double>> W4;
        vector<vector<double>> B4;
        vector<vector<double>> mW1, vW1, mB1, vB1;
        vector<vector<double>> mW2, vW2, mB2, vB2;
        vector<vector<double>> mW3, vW3, mB3, vB3;
        vector<vector<double>> mW4, vW4, mB4, vB4;
        double beta1 = 0.9;
        double beta2 = 0.999;
        double eps = 1e-8;
        int t = 0;
    public:
        Multi_Perceptron()
        : W1(64, vector<double>(4)),
          B1(64, vector<double>(1, 0.0)),
          W2(32, vector<double>(64)),
          B2(32, vector<double>(1, 0.0)),
          W3(16, vector<double>(32)),
          B3(16, vector<double>(1, 0.0)),
          W4(3, vector<double>(16)),
          B4(3, vector<double>(1, 0.0)) {
            this->init();
          }
        void init(){
            random_device rd;
            mt19937 gen(rd());
            double limit_1 = sqrt(2.0 / 64.0); // He 
            normal_distribution<double> dist1(0.0, limit_1);
            for (int i = 0; i < 64; i++){
                for (int j = 0; j < 4; j++){
                    W1[i][j] = dist1(gen);
                }
            }
            double limit_2 = sqrt(2.0 / 32.0);
            normal_distribution<double> dist2(0.0, limit_2);
            for (int i = 0; i < 32; i++)
                for (int j = 0; j < 64; j++)
                    W2[i][j] = dist2(gen);
            double limit_3 = sqrt(2.0 / 16.0);
            normal_distribution<double> dist3(0.0, limit_3);
            for (int i = 0; i < 16; i++)
                for (int j = 0; j < 32; j++)
                    W3[i][j] = dist3(gen);
            double limit_4 = sqrt(6.0 / (16.0 + 3.0));
            normal_distribution<double> dist4(-limit_4, limit_4);
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 16; j++)
                    W4[i][j] = dist4(gen);

            mW1 = vW1 = vector<vector<double>>(64, vector<double>(4, 0.0));
            mB1 = vB1 = vector<vector<double>>(64, vector<double>(1, 0.0));
            mW2 = vW2 = vector<vector<double>>(32, vector<double>(64, 0.0));
            mB2 = vB2 = vector<vector<double>>(32, vector<double>(1, 0.0));
            mW3 = vW3 = vector<vector<double>>(16, vector<double>(32, 0.0));
            mB3 = vB3 = vector<vector<double>>(16, vector<double>(1, 0.0));
            mW4 = vW4 = vector<vector<double>>(3, vector<double>(16, 0.0));
            mB4 = vB4 = vector<vector<double>>(3, vector<double>(1, 0.0));

        }

        void adam_update(vector<vector<double>>& W, vector<vector<double>>& mW, vector<vector<double>>& vW, const vector<vector<double>>& dW, double lr){
            double beta1_t = pow(beta1, t);
            double beta2_t = pow(beta2, t);
            for(size_t i = 0; i < W.size(); i++){
                for(size_t j = 0; j < W[0].size(); j++){
                    mW[i][j] = beta1 * mW[i][j] + (1 - beta1) * dW[i][j];
                    vW[i][j] = beta2 * vW[i][j] + (1 - beta2) * dW[i][j] * dW[i][j];
                    double m_hat = mW[i][j] / (1 - beta1_t);
                    double v_hat = vW[i][j] / (1 - beta2_t);
                    W[i][j] -= lr * m_hat / (sqrt(v_hat) + eps);
                }
            }
        }

        Mat_Ops predict_proba(const vector<vector<double>>& X) const {
            Mat_Ops x(X);
            Mat_Ops Z1 = Mat_Ops(W1).dot(x).add(B1); Z1.relu();
            Mat_Ops Z2 = Mat_Ops(W2).dot(Z1).add(B2); Z2.relu();
            Mat_Ops Z3 = Mat_Ops(W3).dot(Z2).add(B3); Z3.relu();
            Mat_Ops Z4 = Mat_Ops(W4).dot(Z3).add(B4);
            return Z4.stable_exp().softmax_from_exp();
        }

        int count_correct(
            const vector<int>& predictions,
            const vector<int>& labels
        ) {
            if (predictions.size() != labels.size())
                throw runtime_error("Size mismatch in accuracy");

            int correct = 0;
            for (size_t i = 0; i < predictions.size(); i++) {
                if (predictions[i] == labels[i])
                    correct++;
            }
            return correct;
        }

        double cross_entropy_loss(const Mat_Ops& y_true,
                            const Mat_Ops& y_pred) {
            if (y_true.rows() != y_pred.rows() ||
                y_true.cols() != y_pred.cols())
                throw runtime_error("Loss shape mismatch");

            double loss = 0.0;
            int batch_size = y_true.cols();

            for (int j = 0; j < batch_size; j++) {
                for (int i = 0; i < y_true.rows(); i++) {
                    loss += y_true.data()[i][j] *
                            std::log(y_pred.data()[i][j]);
                }
            }

            return -loss / batch_size;
        }

        void fit(const vector<vector<double>>& X, const vector<vector<double>>& Y, int epochs = 100, int batch_size = 64, double lr = 0.01){
            int samples = X[0].size();
            for(int epoch = 0; epoch < epochs; epoch++) {
                int nr_correct = 0;
                // Shuffle indices
                vector<int> perm(samples);
                iota(perm.begin(), perm.end(), 0);
                std::shuffle(perm.begin(), perm.end(), std::mt19937{std::random_device{}()});
                // Shuffle X and Y
                vector<vector<double>> X_shuffled = X;
                vector<vector<double>> Y_shuffled = Y;
                for(int j = 0; j < samples; j++){
                    for(int i = 0; i < X.size(); i++)
                        X_shuffled[i][j] = X[i][perm[j]];
                    for(int i = 0; i < Y.size(); i++)
                        Y_shuffled[i][j] = Y[i][perm[j]];
                }
                for(int start = 0; start < samples; start += batch_size) {
                    t++;
                    int end = min(start + batch_size, samples);
                    // Slice batch
                    vector<vector<double>> x_batch(X.size(), vector<double>(end - start));
                    vector<vector<double>> y_batch(Y.size(), vector<double>(end - start));
                    for(int j = start; j < end; j++){
                        for(int i = 0; i < X.size(); i++)
                            x_batch[i][j - start] = X_shuffled[i][j];
                        for(int i = 0; i < Y.size(); i++)
                            y_batch[i][j - start] = Y_shuffled[i][j];
                    }
                    Mat_Ops x(x_batch);
                    Mat_Ops y_true(y_batch);
                    //Forward
                    Mat_Ops Z1 = Mat_Ops(W1).dot(x).add(B1);
                    Mat_Ops A1 = Z1; A1.relu();
                    Mat_Ops Z2 = Mat_Ops(W2).dot(A1).add(B2);
                    Mat_Ops A2 = Z2; A2.relu();
                    Mat_Ops Z3 = Mat_Ops(W3).dot(A2).add(B3);
                    Mat_Ops A3 = Z3; A3.relu();
                    Mat_Ops Z4 = Mat_Ops(W4).dot(A3).add(B4);
                    Mat_Ops A4 = Z4.stable_exp().softmax_from_exp();
                    //Accuracy
                    auto preds = A4.argmax_axis0();
                    auto labels = y_true.argmax_axis0();
                    for(size_t i = 0; i < preds.size(); i++)
                        if(preds[i] == labels[i]) nr_correct++;

                    //loss & dZ4
                    const double eps = 1e-8;
                    Mat_Ops y_pred = A4.clip(eps);
                    double loss = y_pred.cross_entropy_loss(y_true, y_pred);

                    Mat_Ops dZ4 = y_pred - y_true;

                    // ---------- Backprop ----------
                    int m = x.cols();
                    Mat_Ops dW4 = dZ4.dot(A3.T()) / m;
                    Mat_Ops dB4 = dZ4.sum_axis1() / m;
                    Mat_Ops dA3 = Mat_Ops(W4).T().dot(dZ4);
                    Mat_Ops dZ3 = dA3;
                    for(int i=0;i<dZ3.rows();i++)
                        for(int j=0;j<dZ3.cols();j++)
                            if(Z3.data()[i][j] <= 0) dZ3.data()[i][j] = 0;
                    Mat_Ops dW3 = dZ3.dot(A2.T()) / m;
                    Mat_Ops dB3 = dZ3.sum_axis1() / m;
                    Mat_Ops dA2 = Mat_Ops(W3).T().dot(dZ3);
                    Mat_Ops dZ2 = dA2;
                    for(int i=0;i<dZ2.rows();i++)
                        for(int j=0;j<dZ2.cols();j++)
                            if(Z2.data()[i][j] <= 0) dZ2.data()[i][j] = 0;
                    Mat_Ops dW2 = dZ2.dot(A1.T()) / m;
                    Mat_Ops dB2 = dZ2.sum_axis1() / m;
                    Mat_Ops dA1 = Mat_Ops(W2).T().dot(dZ2);
                    Mat_Ops dZ1 = dA1;
                    for(int i=0;i<dZ1.rows();i++)
                        for(int j=0;j<dZ1.cols();j++)
                            if(Z1.data()[i][j] <= 0) dZ1.data()[i][j] = 0;
                    Mat_Ops dW1 = dZ1.dot(x.T()) / m;
                    Mat_Ops dB1 = dZ1.sum_axis1() / m;
                    // ---------- Update ----------
                    adam_update(W4, mW4, vW4, dW4.data(), lr);
                    adam_update(B4, mB4, vB4, dB4.data(), lr);
                    adam_update(W3, mW3, vW3, dW3.data(), lr);
                    adam_update(B3, mB3, vB3, dB3.data(), lr);
                    adam_update(W2, mW2, vW2, dW2.data(), lr);
                    adam_update(B2, mB2, vB2, dB2.data(), lr);
                    adam_update(W1, mW1, vW1, dW1.data(), lr);
                    adam_update(B1, mB1, vB1, dB1.data(), lr);
                }
                cout << "Epoch " << epoch+1 << "/" << epochs
                    << " Accuracy: " << (100.0 * nr_correct / samples) << "%\n";
            }
        }

        vector<int> predict(const vector<vector<double>>& X) const {
            Mat_Ops x(X);

            // Forward pass
            Mat_Ops Z1 = Mat_Ops(W1).dot(x).add(B1);
            Mat_Ops A1 = Z1; A1.relu();
            Mat_Ops Z2 = Mat_Ops(W2).dot(A1).add(B2);
            Mat_Ops A2 = Z2; A2.relu();
            Mat_Ops Z3 = Mat_Ops(W3).dot(A2).add(B3);
            Mat_Ops A3 = Z3; A3.relu();
            Mat_Ops Z4 = Mat_Ops(W4).dot(A3).add(B4);
            Mat_Ops A4 = Z4.stable_exp().softmax_from_exp();

            // Argmax per column (sample)
            return A4.argmax_axis0();
        }  
};

int main() {
    try {
        auto start = high_resolution_clock::now();
        string link = ""; // <--- Add the file location here 
        FileStream file(link);
        file.getFile();
        file.drop({"Id"});
        PreProcessor prp;
        // Load and preprocess
        auto X_cells = file.exclude({"Species"});
        auto X = prp.toNumericMatrix(X_cells);
        auto Y_cells = file.include({"Species"});
        unordered_map<string, int> label_map;
        auto y_encoded = prp.labelEncode(Y_cells, label_map);
        auto y_onehot = prp.oneHotEncode(y_encoded, label_map.size());
        auto X_scaled = prp.MinMaxScaler(X);
        auto X_t = prp.Transpose(X_scaled);      // (features × samples)
        auto Y_t = prp.Transpose(y_onehot);      // (classes × samples)
        cout << "X shape: "; prp.get_shape(X_t);
        cout << "Y shape: "; prp.get_shape(Y_t);
        Multi_Perceptron model; //Train
        model.fit(X_t, Y_t, 100, 32, 0.01);
        auto predictions = model.predict(X_t); // predict 
        cout << "\nSample predictions:\n";
        vector<int> samples = {1,4,6,10,45,56,78,90,120,134,102};
        for (int i : samples) {
            cout << "Sample " << i
                 << " -> Predicted: " << predictions[i]
                 << " | Actual: " << y_encoded[i] << endl;
        }
        auto end = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(end - start);
        cout << "Execution time: " << duration.count() << " ms" << endl;

    } catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
    }
    return 0;
}


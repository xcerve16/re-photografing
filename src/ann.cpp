#include <vector>
#include <algorithm>
#include <functional>
#include <map>
#include <set>
#include <fstream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/ml/ml.hpp>

typedef std::vector<std::string>::const_iterator vec_iter;

struct ImageData
{
    std::string classname;
    cv::Mat bowFeatures;
};


/**
 * Extract the class name from a file name 
 */
inline std::string getClassName(const std::string& filename)
{
    return filename.substr(filename.find_last_of('/') + 1, 3);
}

/**
 * Extract local features for an image
 */
cv::Mat getDescriptors(const cv::Mat& img)
{
    cv::Ptr<cv::KAZE> kaze = cv::KAZE::create();
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    kaze->detectAndCompute(img, cv::noArray(), keypoints, descriptors);
    return descriptors;
}

/**
 * Read images from a list of file names and returns, for each read image,
 * its class name and its local descriptors
 */
void readImages(vec_iter begin, vec_iter end, std::function<void (const std::string&, const cv::Mat&)> callback)
{
    for (auto it = begin; it != end; ++it)
    {
        std::string filename = *it;
        std::cout << "Reading image " << filename << "..." << std::endl;
        cv::Mat img = cv::imread(filename, 0);
        if (img.empty())
        {
            std::cerr << "WARNING: Could not read image." << std::endl;
            continue;
        }
        std::string classname = getClassName(filename);
        cv::Mat descriptors = getDescriptors(img);
        callback(classname, descriptors);
    }
}

/**
 * Transform a class name into an id
 */
int getClassId(const std::set<std::string>& classes, const std::string& classname)
{
    int index = 0;
    for (auto it = classes.begin(); it != classes.end(); ++it)
    {
        if (*it == classname) break;
        ++index;
    }
    return index;
}

/**
 * Get a binary code associated to a class
 */
cv::Mat getClassCode(const std::set<std::string>& classes, const std::string& classname)
{
    cv::Mat code = cv::Mat::zeros(cv::Size((int)classes.size(), 1), CV_32F);
    int index = getClassId(classes, classname);
    code.at<float>(index) = 1;
    return code;
}

/**
 * Turn local features into a single bag of words histogram of 
 * of visual words (a.k.a., bag of words features)
 */
cv::Mat getBOWFeatures(cv::FlannBasedMatcher& flann, const cv::Mat& descriptors,
                       int vocabularySize)
{
    cv::Mat outputArray = cv::Mat::zeros(cv::Size(vocabularySize, 1), CV_32F);
    std::vector<cv::DMatch> matches;
    flann.match(descriptors, matches);
    for (size_t j = 0; j < matches.size(); j++)
    {
        int visualWord = matches[j].trainIdx;
        outputArray.at<float>(visualWord)++;
    }
    return outputArray;
}

/**
 * Get a trained neural network according to some inputs and outputs
 */
cv::Ptr<cv::ml::ANN_MLP> getTrainedNeuralNetwork(const cv::Mat& trainSamples,
                                                 const cv::Mat& trainResponses)
{
    int networkInputSize = trainSamples.cols;
    int networkOutputSize = trainResponses.cols;
    cv::Ptr<cv::ml::ANN_MLP> mlp = cv::ml::ANN_MLP::create();
    std::vector<int> layerSizes = { networkInputSize, networkInputSize / 2,
                                    networkOutputSize };
    mlp->setLayerSizes(layerSizes);
    mlp->setActivationFunction(cv::ml::ANN_MLP::SIGMOID_SYM);
    mlp->train(trainSamples, cv::ml::ROW_SAMPLE, trainResponses);
    return mlp;
}

/**
 * Receives a column matrix contained the probabilities associated to 
 * each class and returns the id of column which contains the highest
 * probability
 */
int getPredictedClass(const cv::Mat& predictions)
{
    float maxPrediction = predictions.at<float>(0);
    float maxPredictionIndex = 0;
    const float* ptrPredictions = predictions.ptr<float>(0);
    for (int i = 0; i < predictions.cols; i++)
    {
        float prediction = *ptrPredictions++;
        if (prediction > maxPrediction)
        {
            maxPrediction = prediction;
            maxPredictionIndex = i;
        }
    }
    return maxPredictionIndex;
}

/**
 * Get a confusion matrix from a set of test samples and their expected
 * outputs
 */
std::vector<std::vector<int> > getConfusionMatrix(cv::Ptr<cv::ml::ANN_MLP> mlp,
                                                  const cv::Mat& testSamples, const std::vector<int>& testOutputExpected)
{
    cv::Mat testOutput;
    mlp->predict(testSamples, testOutput);
    std::vector<std::vector<int> > confusionMatrix(2, std::vector<int>(2));
    for (int i = 0; i < testOutput.rows; i++)
    {
        int predictedClass = getPredictedClass(testOutput.row(i));
        int expectedClass = testOutputExpected.at(i);
        confusionMatrix[expectedClass][predictedClass]++;
    }
    return confusionMatrix;
}

/**
 * Print a confusion matrix on screen
 */
void printConfusionMatrix(const std::vector<std::vector<int> >& confusionMatrix,
                          const std::set<std::string>& classes)
{
    for (auto it = classes.begin(); it != classes.end(); ++it)
    {
        std::cout << *it << " ";
    }
    std::cout << std::endl;
    for (size_t i = 0; i < confusionMatrix.size(); i++)
    {
        for (size_t j = 0; j < confusionMatrix[i].size(); j++)
        {
            std::cout << confusionMatrix[i][j] << " ";
        }
        std::cout << std::endl;
    }
}

/**
 * Get the accuracy for a model (i.e., percentage of correctly predicted
 * test samples)
 */
float getAccuracy(const std::vector<std::vector<int> >& confusionMatrix)
{
    int hits = 0;
    int total = 0;
    for (size_t i = 0; i < confusionMatrix.size(); i++)
    {
        for (size_t j = 0; j < confusionMatrix.at(i).size(); j++)
        {
            if (i == j) hits += confusionMatrix.at(i).at(j);
            total += confusionMatrix.at(i).at(j);
        }
    }
    return hits / (float)total;
}

/**
 * Save our obtained models (neural network, bag of words vocabulary
 * and class names) to use it later
 */
void saveModels(cv::Ptr<cv::ml::ANN_MLP> mlp, const cv::Mat& vocabulary,
                const std::set<std::string>& classes)
{
    mlp->save("mlp.yaml");
    cv::FileStorage fs("vocabulary.yaml", cv::FileStorage::WRITE);
    fs << "vocabulary" << vocabulary;
    fs.release();
    std::ofstream classesOutput("classes.txt");
    for (auto it = classes.begin(); it != classes.end(); ++it)
    {
        classesOutput << getClassId(classes, *it) << "\t" << *it << std::endl;
    }
    classesOutput.close();
}

int main(int argc, char** argv)
{


    int networkInputSize = 512;
    float trainSplitRatio = 0.7;

    std::cout << "Reading training set..." << std::endl;
    double start = (double)cv::getTickCount();
    std::vector<std::string> files;
    files.push_back("resource/image/ann_cat.jpg");
    files.push_back("resource/image/ann_dog.jpg");

    std::random_shuffle(files.begin(), files.end());

    cv::Mat descriptorsSet;
    std::vector<ImageData*> descriptorsMetadata;
    std::set<std::string> classes;
    readImages(files.begin(), files.begin() + (size_t)(files.size() * trainSplitRatio),
               [&](const std::string& classname, const cv::Mat& descriptors) {
                   // Append to the set of classes
                   classes.insert(classname);
                   // Append to the list of descriptors
                   descriptorsSet.push_back(descriptors);
                   // Append metadata to each extracted feature
                   ImageData* data = new ImageData;
                   data->classname = classname;
                   data->bowFeatures = cv::Mat::zeros(cv::Size(networkInputSize, 1), CV_32F);
                   for (int j = 0; j < descriptors.rows; j++)
                   {
                       descriptorsMetadata.push_back(data);
                   }
               });
    std::cout << "Time elapsed in minutes: " << ((double)cv::getTickCount() - start) / cv::getTickFrequency() / 60.0 << std::endl;

    std::cout << "Creating vocabulary..." << std::endl;
    start = (double)cv::getTickCount();
    cv::Mat labels;
    cv::Mat vocabulary;
    // Use k-means to find k centroids (the words of our vocabulary)
    cv::kmeans(descriptorsSet, networkInputSize, labels, cv::TermCriteria(cv::TermCriteria::EPS +
                                                                          cv::TermCriteria::MAX_ITER, 10, 0.01), 1, cv::KMEANS_PP_CENTERS, vocabulary);
    // No need to keep it on memory anymore
    descriptorsSet.release();
    std::cout << "Time elapsed in minutes: " << ((double)cv::getTickCount() - start) / cv::getTickFrequency() / 60.0 << std::endl;

    // Convert a set of local features for each image in a single descriptors
    // using the bag of words technique
    std::cout << "Getting histograms of visual words..." << std::endl;
    int* ptrLabels = (int*)(labels.data);
    int size = labels.rows * labels.cols;
    for (int i = 0; i < size; i++)
    {
        int label = *ptrLabels++;
        ImageData* data = descriptorsMetadata[i];
        data->bowFeatures.at<float>(label)++;
    }

    // Filling matrixes to be used by the neural network
    std::cout << "Preparing neural network..." << std::endl;
    cv::Mat trainSamples;
    cv::Mat trainResponses;
    std::set<ImageData*> uniqueMetadata(descriptorsMetadata.begin(), descriptorsMetadata.end());
    for (auto it = uniqueMetadata.begin(); it != uniqueMetadata.end(); )
    {
        ImageData* data = *it;
        cv::Mat normalizedHist;
        cv::normalize(data->bowFeatures, normalizedHist, 0, data->bowFeatures.rows, cv::NORM_MINMAX, -1, cv::Mat());
        trainSamples.push_back(normalizedHist);
        trainResponses.push_back(getClassCode(classes, data->classname));
        delete *it; // clear memory
        it++;
    }
    descriptorsMetadata.clear();

    // Training neural network
    std::cout << "Training neural network..." << std::endl;
    start = cv::getTickCount();
    cv::Ptr<cv::ml::ANN_MLP> mlp = getTrainedNeuralNetwork(trainSamples, trainResponses);
    std::cout << "Time elapsed in minutes: " << ((double)cv::getTickCount() - start) / cv::getTickFrequency() / 60.0 << std::endl;

    // We can clear memory now
    trainSamples.release();
    trainResponses.release();

    // Train FLANN
    std::cout << "Training FLANN..." << std::endl;
    start = cv::getTickCount();
    cv::FlannBasedMatcher flann;
    flann.add(vocabulary);
    flann.train();
    std::cout << "Time elapsed in minutes: " << ((double)cv::getTickCount() - start) / cv::getTickFrequency() / 60.0 << std::endl;

    // Reading test set
    std::cout << "Reading test set..." << std::endl;
    start = cv::getTickCount();
    cv::Mat testSamples;
    std::vector<int> testOutputExpected;
    readImages(files.begin() + (size_t)(files.size() * trainSplitRatio), files.end(),
               [&](const std::string& classname, const cv::Mat& descriptors) {
                   // Get histogram of visual words using bag of words technique
                   cv::Mat bowFeatures = getBOWFeatures(flann, descriptors, networkInputSize);
                   cv::normalize(bowFeatures, bowFeatures, 0, bowFeatures.rows, cv::NORM_MINMAX, -1, cv::Mat());
                   testSamples.push_back(bowFeatures);
                   testOutputExpected.push_back(getClassId(classes, classname));
               });
    std::cout << "Time elapsed in minutes: " << ((double)cv::getTickCount() - start) / cv::getTickFrequency() / 60.0 << std::endl;

    // Get confusion matrix of the test set
    std::vector<std::vector<int> > confusionMatrix = getConfusionMatrix(mlp,
                                                                        testSamples, testOutputExpected);

    // Get accuracy of our model
    std::cout << "Confusion matrix: " << std::endl;
    printConfusionMatrix(confusionMatrix, classes);
    std::cout << "Accuracy: " << getAccuracy(confusionMatrix) << std::endl;

    // Save models
    std::cout << "Saving models..." << std::endl;
    saveModels(mlp, vocabulary, classes);

    return 0;
}
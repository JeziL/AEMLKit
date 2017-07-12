//
// Created by Wang Jinli on 2017/6/14.
//

#ifndef AEMLKIT_AEMLKIT_H
#define AEMLKIT_AEMLKIT_H

#include <cstddef>
#include <shark/Models/FFNet.h>

/*!
    @struct TrainingData
    @brief 训练数据
    @field input 特征向量输入
    @field label 目标输出（类别代号: 0, 1, ...）
*/
struct TrainingData {
    double *input;
    unsigned int label;
};

/*!
    @struct FFNetStructure
    @brief 神经网络结构
    @field numInputs 输入神经元数（特征向量维数）
    @field numHiddenLayers 隐藏层数
    @field numClasses 类别数
    @field hiddenLayerSizes 各隐藏层神经元数
*/
struct FFNetStructure {
    size_t numInputs;
    size_t numHiddenLayers;
    size_t numClasses;
    size_t *hiddenLayerSizes;
};

/*!
    @struct PredictResult
    @brief 神经网络预测结果
    @field predictedLabel 预测输出（类别代号: 0, 1, ...）
    @field probabilities 输入分别属于各类别的概率
*/
struct PredictResult {
    unsigned int predictedLabel;
    double *probabilities;
};

/*!
    @typedef ValidatingData
    @brief 验证数据（结构同训练数据）
    @field input 特征向量输入
    @field label 期望输出（类别代号: 0, 1, ...）
*/
typedef TrainingData ValidatingData;


/*!
    @class FeedForwardNet
    @brief 前馈神经网络
*/
class FeedForwardNet {

public:
    /*!
        @var structure 神经网络结构
    */
    FFNetStructure structure;

    /*!
        @brief 构造函数
        @param s 神经网络结构
    */
    FeedForwardNet(FFNetStructure s) { structure = s; };
    /*!
        @brief 构造函数
        @param numInputs 输入神经元数（特征向量维数）
        @param numHiddenLayers 隐藏层数
        @param numClasses 类别数
        @param hiddenLayerSizes 各隐藏层神经元数
     */
    FeedForwardNet(size_t numInputs, size_t numHiddenLayers, size_t numClasses, size_t *hiddenLayerSizes);

    /*!
        @brief 初始化神经网络
    */
    void initFFNet();
    /*!
        @brief 训练神经网络
        @param dataset 训练数据集
        @param count 训练样本数
        @param epochs 训练回合数
     */
    void train(TrainingData *dataset, size_t count, unsigned int epochs);
    /*!
        @brief 验证神经网络训练结果
        @param dataset 验证数据集
        @param count 验证样本数
        @return 被错误分类的样本比例
     */
    double validate(ValidatingData *dataset, size_t count);
    /*!
        @brief 预测未知分类样本
        @param input 特征向量输入
        @return 预测结果
     */
    PredictResult predict(double *input);

private:
    shark::FFNet<shark::LogisticNeuron, shark::LinearNeuron> ffnet;
    shark::ClassificationDataset trainingData;

    shark::ClassificationDataset convertData(TrainingData *dataset, size_t count);

};


#endif //AEMLKIT_AEMLKIT_H

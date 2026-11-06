using UnityEngine;


public class RunInference : MonoBehaviour
{
    [Header("Sentis Setup")]
    public Unity.InferenceEngine.ModelAsset modelAsset;  
    public Texture2D inputTexture; 

    private Unity.InferenceEngine.Model runtimeModel;
    private Unity.InferenceEngine.Worker worker;

    void Start()
    {
        if (modelAsset == null || inputTexture == null)
        {
            Debug.LogError("Asigna el ModelAsset y la Texture2D en el Inspector");
            return;
        }


        runtimeModel = Unity.InferenceEngine.ModelLoader.Load(modelAsset);

        using Unity.InferenceEngine.Tensor<float> inputTensor = Unity.InferenceEngine.TextureConverter.ToTensor(inputTexture, 224, 224, 3);


        worker = new Unity.InferenceEngine.Worker(runtimeModel, Unity.InferenceEngine.BackendType.GPUCompute);

        worker.Schedule(inputTensor);

        using Unity.InferenceEngine.Tensor<float> outputTensor = worker.PeekOutput() as Unity.InferenceEngine.Tensor<float>;
        
        
        using Unity.InferenceEngine.Tensor<float> cpuTensor = outputTensor.ReadbackAndClone();

       
        int maxIndex = 0;
        float maxValue = float.MinValue;

        for (int i = 0; i < cpuTensor.shape.length; i++)
        {
            if (cpuTensor[i] > maxValue)
            {
                maxValue = cpuTensor[i];
                maxIndex = i;
            }
        }

        Debug.Log($"🤖 ¡Inferencia completada con éxito!");
        Debug.Log($"Predicción - Clase detectada (ID): {maxIndex} con valor: {maxValue}");
    }

    void OnDestroy()
    {
       
        worker?.Dispose();
    }
}
#include <fstream> 
#include <iostream> 

#include <NvInfer.h> 
#include <NvOnnxParser.h> 
#include <logger.h> 

using namespace nvinfer1;
using namespace nvonnxparser;
using namespace sample;

int main(int argc, char** argv)
{
	// Create builder 
	Logger m_logger;
	IBuilder* builder = createInferBuilder(m_logger);
	const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
	IBuilderConfig* config = builder->createBuilderConfig();

	// Create model to populate the network 
	INetworkDefinition* network = builder->createNetworkV2(explicitBatch);

	// Parse ONNX file 
	IParser* parser = nvonnxparser::createParser(*network, m_logger);
	bool parser_status = parser->parseFromFile("model.onnx", static_cast<int>(ILogger::Severity::kWARNING));

	// Get the name of network input 
	Dims dim = network->getInput(0)->getDimensions();
	if (dim.d[0] == -1)  // -1 means it is a dynamic model 
	{
		const char* name = network->getInput(0)->getName();
		IOptimizationProfile* profile = builder->createOptimizationProfile();
		profile->setDimensions(name, OptProfileSelector::kMIN, Dims4(1, dim.d[1], dim.d[2], dim.d[3]));
		profile->setDimensions(name, OptProfileSelector::kOPT, Dims4(1, dim.d[1], dim.d[2], dim.d[3]));
		profile->setDimensions(name, OptProfileSelector::kMAX, Dims4(1, dim.d[1], dim.d[2], dim.d[3]));
		config->addOptimizationProfile(profile);
	}


	// Build engine 
	config->setMaxWorkspaceSize(1 << 20);
	ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);

	// Serialize the model to engine file 
	IHostMemory* modelStream{ nullptr };
	assert(engine != nullptr);
	modelStream = engine->serialize();

	std::ofstream p("model.engine", std::ios::binary);
	if (!p) {
		std::cerr << "could not open output file to save model" << std::endl;
		return -1;
	}
	p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
	std::cout << "generate file success!" << std::endl;

	// Release resources 
	modelStream->destroy();
	network->destroy();
	engine->destroy();
	builder->destroy();
	config->destroy();
	return 0;
}

# Project Roadmap

## LLM-Based Smart Contract Generation with Deterministic Outputs

Our mission is to reliably enhance the quality, accuracy, and determinism of large language model (LLM) outputs, specifically tailored for financial services and blockchain smart-contract generation.

----------

### Phase 1: Foundation and Initial Integration

-   Core Technology Development  
	-   Establish the initial LLM pipeline capable of translating natural-language intents into basic smart contracts.  
	-   Implement foundational accuracy measurement methodologies:  
		-   Quantitative benchmarking against public blockchain datasets.  
		-   Basic static analysis integration post-generation.  Integration of static analyzers and linters.
-   Early Data Collection  
	-   Begin systematic collection of initial public datasets (smart contracts, technical documents).  
    -   Initiate private dataset gathering (partner-specified user intents and requirements).  
----------

### Phase 2: Accuracy Enhancement and Extended Integrations 

-  Initial Agentic Capabilities
	-  Integrate foundational agent logic enabling automatic retries and schema-validation loops when initial inference outputs do not meet validation standards.
	-  Prototype early Model Context Protocol (MCP) integration:
		-  Expose key internal tools (such as schema validation and static analysis) through an MCP-compliant interface.
		-  Conduct initial testing of MCP-based tool discovery and invocation within a simplified agentic workflow.
	-  Implement robust logging mechanisms to audit each agent step and MCP interaction.
-   Advanced Accuracy and Determinism  
	-   Deploy advanced pre-inference strategies (synthetic data generation, fine-tuning models).  
	 -   Refine post-inference methods:  
		    -   Enhanced static code analysis.  Filling the gaps found in previous integrations of open-source  static tools.
	-   Minimal expert human validation for critical scenarios.  Estimated use cases, estimated percent of usage. Technical integration of human-in-the-loop.
 	-   Identifying distinct components of smart contracts to assess which parts are suitable for dependable automation and which require continued human oversight
 -   Comprehensive Data Strategy  
	  -   Standardize systematic benchmarking methodology for each use case scenario.  
      -   Build detailed repositories of contract-specific accuracy metrics and binary indicators (automation viability).  
  ----------

### Phase 3: Product Maturity and Broad Market Entry

-   Automation and Security  
	-   Achieve significant automation coverage (clearly defined smart contract scenarios).  
	-   Establish robust guidelines for scenarios requiring human supervision.
 	-   Accuracy prediction for each scenario.    
-   Data-Driven Differentiation (moat)  
	-   Aggregate and continually update specialized contract-generation datasets.  
	-   Implement user feedback loops to refine performance metrics and model configurations.  
      
    

----------

### Phase 4: Scaling and Strategic Partnerships 


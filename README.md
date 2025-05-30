LLM-based Smart Contract Generator
=======================

This repository contains an open-source Web 3.0 project leveraging advanced Large Language Models (LLMs) to transform user intents, expressed in natural language, into executable smart contracts known as Andromeda Digital Objects (ADOs). Accelerate your decentralized app development.

**Note**: The code generated by the LLM is currently of _limited quality_, often containing “hallucinations” or incomplete logic.

Table of Contents
-----------------

1.  [Project Overview](#-project-overview)
        
2.  [Supported Object Types](#-supported-object-types)
   
3.  [Usage](#-usage) 
    
5.  [Generated Examples](#-generated-examples)
    
6.  [Known Limitations](#%EF%B8%8F-known-limitations)
    
7.  [Possible Improvements](#-possible-improvements)
    
8.  [License](#license)
    

📚 Project Overview
----------------

When creating decentralized applications with the smart contract framework, developers regularly need to craft and deploy smart contracts customized to their specific business requirements. This project addresses this challenge by:

*   Accepting a user’s intent in natural language (e.g., “I want an NFT marketplace that charges a commission to sellers”).
    
*   Processing that intent through a Large Language Model.
    
*   Generating an initial smart contract draft matching the described functionality.
    
  

🧩 Supported Object Types
-------------------

This project currently supports generating code for the following object types (based on Andromeda Digital Objects):

*   **marketplace** The Marketplace ADO is a smart contract that allows you to sell your NFTs in a marketplace. 
    
*   **cw20** The CW20 ADO is a smart contract to initiate a standard CW20 token. CW20 is a specification for fungible tokens based on CosmWasm. 
    
*   **auction** The Auction ADO is a smart contract that allows performing custom auctions on NFTs. 
    
*   **cw721** The CW721 ADO is a smart contract to allows users to launch their own custom NFT projects. 
    
*   **timelock** The Timelock ADO or Escrow ADO is a smart contract built to hold funds (Native coins) for a period of time until the set condition is satisfied. 
    
*   **cw20 exchange** The CW20 Exchange ADO is used to sell CW20 tokens for other assets.
  
*   **splitter** The Splitter ADO is a smart contract used to split funds to a preset number of addresses.  
  
*   **crowdfund** A crowdfund contract that allows users to pledge tokens to reach a funding goal.  


💾  Usage
-----

To generate test examples:

```console
python create.py
```

The scripts are located in the /generated directory

📸 Generated examples
-----

![alt text](http://88.198.17.207:1962/static/table.png)
        

⚠️ Known Limitations
-----------------

*   **Hallucinated / Fantasized Code**: The LLM sometimes fabricates or “hallucinates” contract code that may not compile or make logical sense.
    
*   **Incomplete Edge Cases**: Certain flows may be only partially implemented or missing altogether.
    
*   **Deviation from Standards**: The generated code may deviate significantly from known, high-quality templates or standard practices.
    

🚧 Possible Improvements
---------------------

*   **Refined Class Schemas** Structuring clearer class models to guide LLM precision.
    
*   **Price/Recall Matrix** Establishing robust evaluation metrics based on curated gold-standard datasets.
    
*   **Finetuning the LLM** Custom model training with domain-specific data to drastically reduce inaccuracies.
    
*   **Prompt Engineering** Systematic, step-by-step prompt structuring to ensure consistent and reliable outputs.
    
*   **Automated Testing** Incorporating automated compilation and testing processes, improving deployment reliability.
    

License
-------

This project is licensed under the [MIT License](LICENSE.md).

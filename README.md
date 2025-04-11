LLM-based Smart Contract Generator
=======================

This repository hosts an open-source project that uses Large Language Models (LLMs) to transform user intents, described in natural language, into Andromeda Digital Objects. The current system supports seven distinct ADO types—nft\_marketplace, crowdfund, cw20\_exchange, auction\_using\_cw20\_tokens, extended\_marketplace, commission\_based\_sales, and vesting\_and\_staking.

**Note**: The code generated by the LLM is currently of _limited quality_, often containing “fantasized” or incomplete logic.

Table of Contents
-----------------

1.  [Project Overview](#project-overview)
        
3.  [Available Generated Objects Types](#available-generated-objects-types)
    
4.  [Generated Examples](#generated-examples)
    
5.  [Known Limitations](#known-limitations)
    
6.  [Possible Improvements](#possible-improvements)
    
7.  [License](#license)
    

Project Overview
----------------

When building decentralized applications on Andromeda Protocol or similar platforms, developers frequently need to define and deploy — smart contracts tailored to their business needs. This project aims to:

*   Accept a user’s intent in natural language (e.g., “I want an NFT marketplace that charges a commission to sellers”).
    
*   Process that intent through a Large Language Model.
    
*   Output a first draft of an ADO that implements the specified behavior.
    
  

Available Generated Objects Types
-------------------

This project currently supports generating code for the following seven object types:

1.  **nft\_marketplace**An NFT marketplace that supports listing, selling, and buying NFTs.
    
2.  **crowdfund**A crowdfund contract that allows users to pledge tokens to reach a funding goal.
    
3.  **cw20\_exchange**A token exchange platform that swaps or trades native and CW20 tokens.
    
4.  **auction\_using\_cw20\_tokens**A timed or open auction mechanism utilizing CW20 tokens for bidding.
    
5.  **extended\_marketplace**A more feature-rich marketplace, potentially combining aspects of auctions, direct listings, and advanced settlement logic.
    
6.  **commission\_based\_sales**A marketplace or sales mechanism that withholds a commission from each sale for the contract operator.
    
7.  **vesting\_and\_staking**Contracts that manage locked or vested tokens and integrate staking functionality.
    

Generated examples
-----

![alt text](http://88.198.17.207:1962/static/table.png)
[examples](http://88.198.17.207:1962/generate)
        

Known Limitations
-----------------

*   **Hallucinated / Fantasized Code**: The LLM sometimes fabricates or “hallucinates” contract code that may not compile or make logical sense.
    
*   **Incomplete Edge Cases**: Certain flows may be only partially implemented or missing altogether.
    
*   **Low Fidelity to Gold Standards**: The generated ADO may deviate significantly from known, high-quality templates or standard practices.
    

Possible Improvements
---------------------

1.  **Refined Class Schemas**By carefully structuring the prompts and the underlying contract class models, we can guide the LLM toward more accurate generation.
    
2.  **Price/Recall Matrix**Creating a thorough matrix or evaluation scheme around a high-quality gold dataset can give the LLM (or the system orchestrating the LLM) quantitative feedback on how close it is to correct solutions.
    
3.  **Finetuning the LLM**Instead of using a general-purpose LLM, training or finetuning on a curated dataset of ADO examples could dramatically reduce hallucinations and improve fidelity.
    
4.  **Prompt Engineering**Carefully engineering prompts—breaking them down into step-by-step instructions or providing more context—can help reduce randomness in the generated code.
    
5.  **Automated Testing**Creating automated tests or a staging environment where each generated contract is compiled and tested can catch many of the issues before they reach users.
    

We encourage contributors to open pull requests, share prompt engineering insights, or provide improved training data to help make the generator more robust.

License
-------

This project is licensed under the [MIT License](LICENSE.md).

We hope this LLM-based ADO Generator project helps you rapidly prototype ADOs for CosmWasm and beyond. We welcome criticism, feature requests, and pull requests—together, we can improve how decentralized applications get built!

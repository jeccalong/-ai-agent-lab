# Python LangChain AI Agent - Student Lab Instructions

## Prerequisites

Before starting this lab, ensure you have the following installed:

1. **Python (Version 3.9 or higher)**
   - Download from: https://www.python.org/downloads/
   - Verify installation: `python --version` or `python3 --version`
   - Verify pip: `pip --version` or `pip3 --version`

2. **Visual Studio Code**
   - Install the "Python" extension from the VS Code marketplace
   - Install the "GitHub Copilot" extension

3. **GitHub Models Access**
   - Sign in to GitHub with your account
   - Visit https://github.com/marketplace/models
   - Find and select the **gpt-4o** model from OpenAI
   - Click "Get started" or "Deploy" to enable the model for your account
   - Create a Personal Access Token (PAT) with the following steps:
     1. **Navigate to Developer Settings**: In GitHub, click your profile photo, go to Settings, then click Developer settings in the left sidebar
     2. **Select Fine-grained Tokens**: Under "Personal access tokens", select Fine-grained tokens and then click Generate new token
     3. **Configure Token Details**: Give the token a descriptive name (e.g., "GitHub Models Access") and set an expiration period (recommended for security)
     4. **Select Repository Access**: Choose whether the token can access all repositories or only specific ones. For security, it is best to select "Only select repositories" and choose the minimal number needed
     5. **Add Permissions**: Under the "Permissions" section, find the **Models** permission under "Account permissions" and set its access level to **Read**
     6. **Generate and Save**: Click Generate token at the bottom of the page. Immediately copy the token and store it in a secure location, as you will not be able to see it again
   - Or use your GitHub Copilot subscription which includes access to GitHub Models

4. **Create .gitignore File**
   - Download the Python .gitignore template from GitHub:
     - Visit: https://github.com/github/gitignore/blob/main/Python.gitignore
     - Click "Raw" button and save the content to a `.gitignore` file in your project root
     - Or use this direct link: https://raw.githubusercontent.com/github/gitignore/main/Python.gitignore
   - **Important**: Verify that `.env` is included in the .gitignore (it should be by default)
   - This prevents committing sensitive data (like your GitHub token) and generated files to version control

5. **Create .env File**
   - Create a `.env` file in the project root directory (this should already be in your .gitignore)
   - Add your GitHub token: `GITHUB_TOKEN=your_token_here`
   - Replace `your_token_here` with your actual GitHub Personal Access Token
   - **Important**: Never commit this file to version control - verify it's listed in `.gitignore`

---

## 💡 Working with GitHub Copilot Chat for Debugging

**When you encounter runtime errors:**

1. **Copy the error message** from your terminal or console output
2. **Open GitHub Copilot Chat** (click the chat icon in VS Code sidebar or use `Ctrl+Alt+I` / `Cmd+Alt+I`)
3. **Provide context** by typing something like:
   ```
   I'm getting this error when running my Python LangChain application:
   [paste your error message here]
   
   Can you help me fix it?
   ```
4. **Review the suggestion** and apply the fix Copilot recommends
5. **Ask follow-up questions** if you don't understand the solution

**Pro Tips:**
- Include the full error stack trace for better diagnostics
- Mention what you were trying to do when the error occurred
- If the first suggestion doesn't work, tell Copilot and ask for alternatives
- Use Copilot Chat to explain error messages you don't understand

---

## Lab Exercise: Building an AI Agent with Python and LangChain

### Part 1: Project Setup

**Prompt 1: Create Requirements File**
```
Create a requirements.txt file for a Python LangChain AI agent application that includes:
- langchain
- langchain-openai
- langchain-core
- python-dotenv
```

**Prompt 2: Create Virtual Environment and Install Dependencies**
```
Create a Python virtual environment named 'venv' and provide the commands to:
- Create the virtual environment
- Activate it (show commands for both Windows and Linux/Mac)
- Install the dependencies from requirements.txt
```

**Prompt 3: Create Basic App Structure**
```
Create an app.py file that:
- Has the necessary imports for dotenv
- Has a main() function
- Calls main() when the script is run directly (using if __name__ == "__main__")
- Includes a starting message with emoji
```

**Prompt 4: Create VS Code Configuration**
```
Create VS Code tasks.json and launch.json for the python console application in the python-langchain folder.
```

---

### Part 2: Basic Application Setup (Without Tools)

**Prompt 5: Load Environment Variables**
```
In app.py, add code to:
- Import and load environment variables using dotenv
- Check if GITHUB_TOKEN exists in environment variables
- Display an error message with helpful instructions if the token is not found
- Return early if no token is found
- Include helpful user feedback with emoji
```

**Prompt 6: Initialize ChatOpenAI Model**
```
Add code to:
- Import ChatOpenAI from langchain_openai
- Create a ChatOpenAI instance that uses:
  - model="openai/gpt-4o"
  - temperature=0 for deterministic responses
  - base_url="https://models.github.ai/inference"
  - api_key from the GITHUB_TOKEN environment variable
```

**Prompt 7: Test Basic Query (Without Tools)**
```
Add code to:
- Import necessary classes for creating a simple agent
- Create a test query: "What is 25 * 4 + 10?"
- Call llm.invoke() with the query
- Print the response content
- Note: The AI will try to answer on its own without tools
```

**Test Point**: Run the application with `python app.py`. You should see the AI attempt to answer the math question, but it may not be accurate since it doesn't have access to calculation tools.

---

### Part 3: Adding Tools and Agent

**Prompt 8: Import Agent and Tool Classes**
```
Update the imports to include:
- create_agent from langchain.agents
- Tool from langchain_core.tools
- datetime from the datetime module
```

**Prompt 9: Create Calculator Tool Function**
```
Before the main() function, create a calculator function that:
- Takes a string input parameter
- Has a docstring explaining it evaluates mathematical expressions
- Uses Python's eval() to evaluate the expression (with safety considerations for demo purposes)
- Includes try-except error handling
- Returns the result as a string
```

**Prompt 10: Create Tools List with Calculator**
```
After initializing the LLM in main(), create a tools list with:
- A Tool object for the calculator
- Use name="Calculator"
- Use the calculator function as func
- Include a detailed description explaining when to use it
```

**Prompt 11: Create Agent with Calculator Tool**
```
Add code to:
- Create an agent using create_agent()
- Pass the llm and tools list
- Set debug=True for verbose output
- Create a test query: "What is 25 * 4 + 10?"
- Use agent_executor.invoke() with a dictionary containing the input query
- Print the result['output']
- Wrap in try-except for error handling
```

**Test Point**: Run the application again. Now the AI should use the Calculator tool to accurately calculate "What is 25 * 4 + 10?" and return 110. You should see debug output showing the tool being called.

---

**Prompt 12: Test Time Query (Without Tool)**
```
Replace the math query with a new query: "What time is it right now?"
Comment out the Calculator tool from the tools list
Run the application and observe that the AI cannot provide the current time accurately
```

**Test Point**: Run the application. The AI will not know the current time since it doesn't have access to system time.

---

**Prompt 13: Create Time Tool Function**
```
Before the main() function, create a get_current_time function that:
- Takes a string input parameter (required by Tool interface)
- Has a docstring explaining it returns the current date and time
- Uses datetime.now().strftime("%Y-%m-%d %H:%M:%S")
- Returns the formatted string
```

**Prompt 14: Add Time Tool to Tools List**
```
Add a Tool object to the tools list that:
- Has name="get_current_time"
- Uses the get_current_time function as func
- Has a description explaining when to use it
- Keep the Calculator tool in the list as well
```

**Test Point**: Run the application. Now the AI should use the get_current_time tool to provide the current time.

---

**Prompt 15: Test String Query (Without Tool)**
```
Replace the query with: "Reverse the string 'Hello World'"
Comment out both the Calculator and Time tools
Run the application and observe that the AI attempts to reverse the string but may not be reliable
```

**Test Point**: Run the application. The AI will try to reverse the string on its own, which may not be perfect.

---

**Prompt 16: Create String Reversal Tool Function**
```
Before the main() function, create a reverse_string function that:
- Takes a string input parameter
- Has a docstring explaining it reverses a string
- Returns the reversed string using Python slice notation [::-1]
```

**Prompt 17: Add String Tool to Tools List**
```
Add a Tool object to the tools list that:
- Has name="reverse_string"
- Uses the reverse_string function as func
- Has a description: "Reverses a string. Input should be a single string."
- Ensure all three tools are uncommented (Calculator, get_current_time, reverse_string)
```

**Test Point**: Run the application. The AI should use the reverse_string tool to accurately reverse the string and return "dlroW olleH".

---

**Prompt 18: Create Multiple Test Queries**
```
Replace the single query with a list of test queries:
- "What time is it right now?"
- "What is 25 * 4 + 10?"
- "Reverse the string 'Hello World'"

Add a for loop that:
- Prints "Running example queries:" before the loop
- Iterates through each query
- Prints the query with formatting (📝 emoji and separator line)
- Calls agent_executor.invoke() for each query
- Prints the result with formatting (✅ emoji)
- Includes try-catch error handling for each query (❌ emoji for errors)
```

**Test Point**: Run the application. All three queries should now work accurately using their respective tools. You should see debug output showing which tool is being called for each query.

---

**Prompt 19: Improve Output Formatting**
```
Update the output formatting to:
- Use print("─" * 50) for separator lines
- Add newlines for better spacing between queries
- Add a completion message at the end: "🎉 Agent demo complete!"
```

---

**Prompt 20: Add System Message**
```
Update the agent initialization to include a system message that instructs the AI to be professional and succinct. Add this configuration when creating the agent.
```

---

### Part 4: Final Testing

**Testing Instructions:**

Run the application with all three tools enabled. You should observe:
- Time queries return accurate current time via get_current_time tool
- Math calculations are precise via Calculator tool
- String reversal is correct via reverse_string tool
- The AI seamlessly chooses and uses the appropriate tool for each query
- Debug output shows the agent's reasoning and tool selection

---

## Expected Output

When running with all tools enabled and debug mode, you should see output similar to:

```
🤖 Python LangChain Agent Starting...

Running example queries:


📝 Query: What time is it right now?
──────────────────────────────────────────────────

[Agent reasoning and tool calls displayed in debug mode]

✅ Result: The current date and time is 2026-01-12 14:30:45


📝 Query: What is 25 * 4 + 10?
──────────────────────────────────────────────────

[Agent reasoning and tool calls displayed in debug mode]

✅ Result: 110


📝 Query: Reverse the string 'Hello World'
──────────────────────────────────────────────────

[Agent reasoning and tool calls displayed in debug mode]

✅ Result: dlroW olleH


🎉 Agent demo complete!
```

---

## Final Assessment

Test your understanding by completing this assessment:

### Weather Tool with Multi-Function Calling

**Objective:** Create a Weather Tool that demonstrates the AI's ability to chain multiple function calls together.

**Requirements:**

1. **Create a Weather Tool function** that:
   - Takes a string input parameter (date in format "YYYY-MM-DD")
   - Has a docstring explaining it returns weather information for a given date
   - Returns "Sunny, 72°F" if the date matches today's date (use datetime.now().strftime("%Y-%m-%d"))
   - Returns "Rainy, 55°F" for all other dates
   - Include proper error handling

2. **Add the tool** to your tools list:
   - Create a Tool object with name="get_weather"
   - Use the weather function as func
   - Include a detailed description explaining it accepts a date parameter (formatted as "YYYY-MM-DD")
   - Keep the get_current_time tool in the tools array as well

3. **Test with a query** that requires two function calls:
   - Ask the AI: "What's the weather like today?"
   - The AI should:
     - First call get_current_time() to get today's date
     - Then call get_weather with that date to get the weather
     - Return a complete answer combining both pieces of information

**Success Criteria:**
- The AI successfully chains two function calls without explicit instruction
- Mock weather data is returned based on the current date
- The response is coherent and answers the original question

---

## Discussion Questions

After completing the lab, consider:

1. What's the difference between responses with and without tools?
2. Why are tools important for AI agents?
3. How does LangChain orchestrate the tool calls?
4. What is the role of the create_agent function in this architecture?
5. How does Python's simplicity compare to JavaScript and Java for building AI agents?

---

## Extension Challenges

If you finish early, try:

### Additional Tools
1. Create a WeatherTool function that returns mock weather data for a given city
2. Add a FileTool that can read/write text files using Python's built-in file operations
3. Create a WebSearchTool that simulates web search results
4. Add conversation memory to maintain context across multiple interactions using ConversationBufferMemory

### Cross-Cutting Concerns (Use Copilot to Help!)

**Logging & Observability**
5. Add comprehensive logging throughout the application
   - Log all AI requests and responses
   - Log tool calls with parameters and results
   - Use Python's logging module with different log levels
   - Include timestamps and log levels (INFO, DEBUG, ERROR)
   - Ask Copilot: "Add logging using Python's logging module to track AI interactions and tool calls"

**Performance Monitoring**
6. Add performance metrics and timing
   - Measure response time for each AI query
   - Track tool execution duration using time.time()
   - Log slow queries (over a threshold)
   - Ask Copilot: "Add performance monitoring to track query response times"

**Error Handling & Resilience**
7. Implement robust error handling
   - Add retry logic with exponential backoff for API failures
   - Handle rate limiting scenarios gracefully
   - Provide detailed error messages to users
   - Use custom exception classes
   - Ask Copilot: "Add retry logic with exponential backoff for AI API calls"

**Input Validation**
8. Add input validation and sanitization
   - Validate user queries before sending to AI
   - Sanitize inputs to prevent injection attacks
   - Set maximum query length limits
   - Use type hints for better code clarity
   - Ask Copilot: "Add input validation to sanitize and validate user queries"

**Security Improvements**
9. Improve the calculator function's security
   - Replace eval() with a safer alternative like ast.literal_eval or a math parser library
   - Implement whitelist of allowed operations
   - Add input validation for mathematical expressions
   - Ask Copilot: "Replace eval() with a safer method for evaluating mathematical expressions"

**Handling Rate Limit Responses (HTTP 429)**
10. Handle rate limiting responses from the API
    - Detect and catch HTTP 429 (Too Many Requests) errors
    - Parse retry-after headers from the response
    - Implement automatic retry after the specified delay
    - Display user-friendly messages when rate limited
    - Ask Copilot: "Add handling for HTTP 429 rate limit responses with automatic retry"

---

## Key Differences from Java and JavaScript

As you work through this lab, you may notice some key differences compared to other language implementations:

1. **Virtual Environments**: Python uses virtual environments (venv) for dependency isolation, while JavaScript uses node_modules and Java uses Maven
2. **Simplicity**: Python's syntax is often more concise—string reversal is `[::-1]` vs JavaScript's `split("").reverse().join("")`
3. **Tool Definition**: Python uses simple functions with docstrings, while Java uses annotations and JavaScript uses object literals
4. **Type System**: Python is dynamically typed (though type hints are available), while Java is statically typed and JavaScript/TypeScript can be either
5. **Agent Creation**: Python uses `create_agent()`, JavaScript uses `initializeAgentExecutorWithOptions()`, Java uses Semantic Kernel's plugin system

All three frameworks accomplish the same goal—building AI agents with tool calling—but leverage the strengths and idioms of their respective languages.

---

## Troubleshooting

**Common Issues:**

1. **"ModuleNotFoundError"**: Make sure your virtual environment is activated and you ran `pip install -r requirements.txt`
2. **"GITHUB_TOKEN not found"**: Verify your .env file is in the correct directory and contains the token
3. **Virtual environment activation issues**: 
   - Windows: `venv\Scripts\activate`
   - Linux/Mac: `source venv/bin/activate`
4. **API errors**: Check that your GitHub token is valid and has proper permissions
5. **eval() security warnings**: This is expected for demo purposes; see Extension Challenge #9 for safer alternatives

**Getting Help:**
- Use GitHub Copilot Chat to explain errors
- Check the LangChain documentation: https://python.langchain.com/
- Review the GitHub Models documentation: https://github.com/marketplace/models
- Make sure you're using Python 3.9 or higher

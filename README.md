# LLM Comparison Toolkit

A Python tool for comparing responses from multiple Large Language Model APIs side by side.

## Features

- Compares responses from three AI APIs:
  - Inception Labs API
  - OpenAI API (GPT-3.5-turbo)
  - Google Gemini AI
- Measures and records response times for each API
- Saves results in CSV format for easy analysis
- Reads prompts from external text files
- Handles API errors gracefully

## Installation

### Prerequisites
- Python 3.7 or higher
- pip (Python package installer)

### Step-by-Step Installation

1. **Clone or download this repository:**
   ```bash
   git clone <repository-url>
   cd LLM-comparison-toolkit
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv

   # Activate virtual environment
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**
   ```bash
   # Copy the example environment file
   cp .env.example .env

   # Edit .env file with your actual API keys
   nano .env  # or use any text editor
   ```

5. **Get your API Keys:**
   - **OpenAI**:
     - Visit [OpenAI Platform](https://platform.openai.com/api-keys)
     - Create an account and generate an API key
     - Add to `.env` as: `OPENAI_API_KEY=sk-your-key-here`

   - **Google Gemini**:
     - Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
     - Create a Google account and generate an API key
     - Add to `.env` as: `GEMINI_API_KEY=your-key-here`

   - **Inception Labs**:
     - Contact Inception Labs for API access
     - Add to `.env` as: `INCEPTION_API_KEY=your-key-here`
     - Note: You may need to adjust the API endpoint in the code

## How to Run

### Basic Usage

1. **Prepare your prompt:**
   ```bash
   # Edit the prompt file with your question
   echo "What are the benefits of renewable energy?" > prompt.txt
   ```

2. **Run the comparison:**
   ```bash
   python llm_comparison.py
   ```

3. **Check the results:**
   - Console output shows a summary
   - Detailed results saved to `llm_comparison_results.csv`

### Example Run

```bash
$ python llm_comparison.py
Starting LLM Comparison...
Prompt: What are the benefits of renewable energy?

Calling APIs...
- Calling Inception Labs API...
- Calling OpenAI API...
- Calling Gemini AI...

Results saved to: llm_comparison_results.csv

================================================================================
COMPARISON SUMMARY
================================================================================
Prompt: What are the benefits of renewable energy?

Inception Labs (2.341s):
  Success: True
  Response: Renewable energy offers numerous benefits including reduced carbon emissions, energy independence, job creation...

OpenAI (1.823s):
  Success: True
  Response: Renewable energy sources like solar, wind, and hydroelectric power provide several key advantages...

Gemini AI (1.456s):
  Success: True
  Response: The transition to renewable energy brings multiple benefits for both the environment and economy...
```

## Understanding the Results

### Console Output
The script provides real-time feedback and a summary including:
- **Response Time**: How long each API took to respond (in seconds)
- **Success Status**: Whether the API call was successful
- **Response Preview**: First 200 characters of each response

### CSV File Structure
The `llm_comparison_results.csv` file contains two rows:

**Row 1 - Main Results:**
| Column | Description |
|--------|-------------|
| Timestamp | When the comparison was run |
| Prompt | The exact prompt sent to all APIs |
| Inception_Response | Full response from Inception Labs |
| Inception_Time_Seconds | Response time in seconds |
| Inception_Success | True/False for API success |
| OpenAI_Response | Full response from OpenAI |
| OpenAI_Time_Seconds | Response time in seconds |
| OpenAI_Success | True/False for API success |
| Gemini_Response | Full response from Gemini |
| Gemini_Time_Seconds | Response time in seconds |
| Gemini_Success | True/False for API success |

**Row 2 - Timing Summary:**
A quick reference row showing just the response times for easy comparison.

### Analyzing Results

1. **Response Quality**: Compare the actual responses to see which API provides the most helpful, accurate, or detailed answer for your use case.

2. **Speed Comparison**: Check the timing columns to see which API responds fastest. Typical ranges:
   - **Fast**: < 2 seconds
   - **Moderate**: 2-5 seconds
   - **Slow**: > 5 seconds

3. **Reliability**: Check the Success columns. Failed calls might indicate:
   - API rate limits
   - Network issues
   - Invalid API keys
   - Service outages

4. **Cost Analysis**: Different APIs have different pricing models. Faster responses might cost more per request.

### Example CSV Output
```csv
Timestamp,Prompt,Inception_Response,Inception_Time_Seconds,Inception_Success,OpenAI_Response,OpenAI_Time_Seconds,OpenAI_Success,Gemini_Response,Gemini_Time_Seconds,Gemini_Success
2024-01-15 14:30:22,"What are the benefits of renewable energy?","Renewable energy offers numerous benefits...",2.341,True,"Renewable energy sources like solar...",1.823,True,"The transition to renewable energy...",1.456,True
2024-01-15 14:30:22 - Timing Summary,Response Times (seconds),"Inception: 2.341s",,,"OpenAI: 1.823s",,,"Gemini: 1.456s",,
```

## Customization

- **Models**: Edit the model names in the script (e.g., change from `gpt-3.5-turbo` to `gpt-4`)
- **Parameters**: Adjust temperature, max_tokens, etc. in each API call function
- **Inception Labs Endpoint**: Update the `inception_endpoint` URL if using a different endpoint

## Error Handling

The tool handles various error scenarios:
- Missing API keys
- Network timeouts
- API rate limits
- Invalid responses

All errors are logged in the CSV output for analysis.
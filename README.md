# Local LLM Circuit Visualizer

A "microscope" for seeing how AI thinks—running entirely on your computer.

![App Screenshot](https://via.placeholder.com/1200x800?text=Local+LLM+Circuit+Visualizer)

## What is this?

You know how MRI scans let doctors see inside a brain without opening it up? This tool does the same for Artificial Intelligence models.

When an AI writes text, it breaks words down into numbers and passes them through layers of math. This app visualizes that process, letting you see:
- **Attention**: How the AI "looks back" at previous words to decide what to say next.
- **Circuits**: Patterns in the AI's "brain" that do specific jobs, like copying names or tracking grammar.
- **Predictions**: What the AI was thinking of saying at each step.

Best of all, **it's 100% private**. The AI runs locally on your machine, so your data never leaves your computer.

## Features

- **See Attention**: Watch the AI connect words (e.g., seeing how "Potter" makes it think of "Harry").
- **Circuit Detection**: Automatically finds "Induction Heads" (patterns that help the AI learn from context).
- **Logit Lens**: Peek at the AI's top guesses for the next word.
- **Offline & Private**: Downloads models once, then runs offline.

## Getting Started

### Prerequisites

You need to have [Node.js](https://nodejs.org/) installed on your computer.

### Installation

1. Clone or download this folder.
2. Open a terminal (Command Prompt on Windows, Terminal on Mac).
3. Navigate to the folder:
   ```bash
   cd local-llm-circuit-visualizer
   ```
4. Install dependencies:
   ```bash
   npm install
   ```

### Running the App

1. Start the application:
   ```bash
   npm run tauri dev
   ```
2. The app window should appear!

### How to Use

1. **Load a Model**: Click "DistilGPT-2" (recommended for beginners). It will download (~50MB) the first time.
2. **Enter a Prompt**: Type something like "The quick brown fox" or "Harry Potter and the".
3. **Click Run**: Watch the magic happen!
4. **Explore**:
   - Hover over the colored squares (the **Heatmap**) to see how words connect.
   - Use the **Layer** and **Head** dropdowns to explore different parts of the AI's brain.
   - Check the **Insights** panel on the right for plain-English explanations.

## License

MIT License

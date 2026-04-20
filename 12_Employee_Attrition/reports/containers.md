## What is devcontainer file?

The `devcontainer.json` file is a configuration file used with Visual Studio Code's **Dev Containers** extension to define a development environment inside a Docker container. This setup allows for a consistent, isolated, and reproducible development environment that can be shared across teams. It’s particularly useful for projects like the Employee Attrition Prediction project, where you need to ensure all developers have the same dependencies, tools, and environment settings.

Here’s a breakdown of the current `devcontainer.json` file:

- **Purpose**: It configures a development container using an existing Dockerfile (`Dockerfile.api`) for the project.
- **Key Settings**:
  - `"name": "Existing Dockerfile"`: Names the dev container.
  - `"build"`:
    - `"context": ".."`: Sets the build context to the parent directory of `.devcontainer/`, which is the project root.
    - `"dockerfile": "../Dockerfile.api"`: Specifies the Dockerfile to use for building the container (in this case, the one for the FastAPI real-time API).
  - Other sections (e.g., `features`, `forwardPorts`, `postCreateCommand`, `customizations`, `remoteUser`) are commented out but can be used to add features, forward ports, run post-creation commands, customize tools, or set a non-root user.

This file is currently set up for a generic development environment using `Dockerfile.api`. However, for the Employee Attrition Prediction project, we can enhance it to better suit the project’s needs, such as forwarding the FastAPI port, installing additional tools, and setting up a non-root user for security.

---

### Updated `devcontainer.json` for the Employee Attrition Project

I’ll update the `devcontainer.json` file to align with the Employee Attrition Prediction project’s requirements as of 11:30 PM -04 on Sunday, May 18, 2025. The updated version will:

1. Use `Dockerfile.api` as the base (since it’s already specified and suits the FastAPI development needs).
2. Forward port 8000 for the FastAPI API.
3. Add features for Python development (e.g., Python interpreter, linting tools).
4. Run post-creation commands to install dependencies and set up the environment.
5. Configure VS Code customizations for Python development (e.g., linting, formatting).
6. Set a non-root user for better security.
7. Ensure compatibility with the project’s folder structure and dependencies (e.g., `uv` for package management).

Here’s the updated `devcontainer.json`:

```jsonc
// For format details, see https://aka.ms/devcontainer.json. For config options, see the
// README at: https://github.com/devcontainers/templates/tree/main/src/docker-existing-dockerfile
{
	"name": "Employee Attrition Dev Environment",
	"build": {
		// Sets the run context to the project root (one level up from .devcontainer folder).
		"context": "..",
		// Use the Dockerfile.api for the FastAPI real-time API.
		"dockerfile": "../Dockerfile.api"
	},

	// Features to add to the dev container. More info: https://containers.dev/features.
	"features": {
		// Add Python development tools.
		"ghcr.io/devcontainers/features/python:1": {
			"version": "3.9"
		},
		// Add Git for version control.
		"ghcr.io/devcontainers/features/git:1": {}
	},

	// Forward port 8000 for FastAPI API development.
	"forwardPorts": [8000],

	// Run commands after the container is created to set up the environment.
	"postCreateCommand": "uv sync && echo 'Environment setup complete!'",

	// Configure tool-specific properties for VS Code.
	"customizations": {
		"vscode": {
			// Extensions to install for development.
			"extensions": [
				"ms-python.python", // Python extension for IntelliSense, linting, debugging
				"ms-python.vscode-pylance", // Enhanced Python language support
				"ms-python.black-formatter", // Black formatter for Python code
				"ms-python.isort", // isort for sorting imports
				"ms-vscode.live-server", // For previewing index.html
				"GitHub.copilot" // GitHub Copilot for code suggestions
			],
			// VS Code settings for Python development.
			"settings": {
				"python.defaultInterpreterPath": "/usr/local/bin/python",
				"python.linting.enabled": true,
				"python.linting.pylintEnabled": true,
				"python.formatting.provider": "black",
				"editor.formatOnSave": true,
				"editor.defaultFormatter": "ms-python.black-formatter",
				"python.sortImports": true,
				"terminal.integrated.defaultProfile.linux": "bash"
			}
		}
	},

	// Connect as a non-root user for better security.
	"remoteUser": "vscode"
}
```

---

### Explanation of Updates

1. **Name**:
   - Changed to `"Employee Attrition Dev Environment"` to reflect the project’s purpose.

2. **Build**:
   - Kept the existing `context` and `dockerfile` settings, as `Dockerfile.api` is appropriate for developing the FastAPI real-time API (`app.py`).

3. **Features**:
   - Added the `python:1` feature to ensure Python 3.9 is installed (matching the `Dockerfile.api` base image of `python:3.9-slim`).
   - Added the `git:1` feature for version control, as the project uses Git and GitHub Actions for CI/CD.

4. **ForwardPorts**:
   - Forwarded port `8000`, which is the default port for the FastAPI application (`uvicorn` runs on port 8000 in `Dockerfile.api`).

5. **PostCreateCommand**:
   - Added `uv sync` to install dependencies using `uv` (as specified in the project’s `pyproject.toml` and `uv.lock`).
   - Included a confirmation message to verify setup completion.

6. **Customizations**:
   - **VS Code Extensions**:
     - Added Python-related extensions (`ms-python.python`, `ms-python.vscode-pylance`) for IntelliSense, linting, and debugging.
     - Added formatting tools (`ms-python.black-formatter`, `ms-python.isort`) to enforce the project’s code style (Black and isort are listed in `pyproject.toml` dev-dependencies).
     - Added `ms-vscode.live-server` for previewing `index.html` (the web interface for the FastAPI API).
     - Added `GitHub.copilot` for enhanced coding assistance.
   - **VS Code Settings**:
     - Set the Python interpreter path to match the container’s Python installation.
     - Enabled linting with Pylint and formatting with Black, aligning with the project’s dev-dependencies.
     - Enabled format-on-save and import sorting for consistency.
     - Set the default terminal to `bash` for better compatibility with Linux-based containers.

7. **RemoteUser**:
   - Set to `vscode`, a non-root user commonly used in dev containers for security (prevents running as root inside the container).

---

### Why These Changes?

- **Project-Specific Needs**:
  - The Employee Attrition project uses Python 3.9 (as specified in `Dockerfile.api`), FastAPI for the real-time API, and `uv` for package management. The updated `devcontainer.json` ensures these dependencies are properly set up.
  - Port 8000 is forwarded to allow local testing of the FastAPI API (`http://localhost:8000`).
  - The project’s `pyproject.toml` includes dev-dependencies like `black` and `isort`, so the VS Code settings enforce these tools for consistent code formatting.

- **Developer Experience**:
  - The added VS Code extensions and settings streamline development by providing linting, formatting, and debugging support.
  - The `postCreateCommand` ensures the environment is ready to use immediately after the container is created.

- **Security**:
  - Using a non-root user (`vscode`) reduces the risk of unintended file permission issues or security vulnerabilities during development.

---

### How to Use the Updated `devcontainer.json`

1. **Prerequisites**:
   - Install Visual Studio Code and the Dev Containers extension.
   - Ensure Docker is installed and running on your machine.

2. **Place the File**:
   - The `devcontainer.json` file should already be in the `.devcontainer/` directory of your project (as per the folder structure).

3. **Open in VS Code**:
   - Open the project folder in VS Code.
   - Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on macOS) to open the command palette.
   - Select **Dev Containers: Reopen in Container**.
   - VS Code will build the container using `Dockerfile.api`, apply the settings, and set up the environment.

4. **Verify Setup**:
   - Open a terminal in VS Code (should default to `bash`).
   - Run `python --version` to confirm Python 3.9 is installed.
   - Run `uv --version` to confirm `uv` is available.
   - Start the FastAPI app: `uvicorn app:app --host 0.0.0.0 --port 8000`.
   - Access `http://localhost:8000` in your browser to test the API.

---

### Additional Notes

- **Dockerfile.api Compatibility**:
  - The `Dockerfile.api` already installs `uv` and syncs dependencies, so the `postCreateCommand` leverages this setup. If you modify `Dockerfile.api` (e.g., change the base image), ensure the Python version in the `features` section matches.

- **Future Enhancements**:
  - If you need additional tools (e.g., AWS CLI for interacting with ECS), you can add them via the `features` section (e.g., `"ghcr.io/devcontainers/features/aws-cli:1": {}`).

---

To create a separate `devcontainer.json` file for the `Dockerfile.batch` in your Employee Attrition Prediction project, you can follow a similar structure to the one provided for `Dockerfile.api`, but tailor it to the batch prediction use case. The `Dockerfile.batch` is designed for running the batch prediction script (`predict_model.py`), which processes employee data in bulk and saves results to a file (e.g., `data/predictions/predictions.csv`). This environment should focus on Python development, data processing tools, and compatibility with the project’s dependencies managed by `uv`.

Here’s how you can create and configure the new `devcontainer.json` for `Dockerfile.batch`:

### Steps to Create a Separate `devcontainer.json` for `Dockerfile.batch`

1. **Locate or Create the `.devcontainer` Directory**:
   - Ensure the `.devcontainer/` directory exists in your project root (e.g., `12_Employee_Attrition/.devcontainer/`). If it doesn’t, create it.

2. **Create a New File**:
   - Inside `.devcontainer/`, create a file named `devcontainer-batch.json` to distinguish it from the API-specific `devcontainer.json`.

3. **Configure the File**:
   - Update the configuration to use `Dockerfile.batch`, forward any relevant ports (if needed), and set up the environment for batch processing.

### Updated `devcontainer-batch.json` for `Dockerfile.batch`

Below is the tailored `devcontainer-batch.json` suited for the Employee Attrition Prediction project’s batch prediction environment as of 12:27 AM -04 on Monday, May 19, 2025:

```jsonc
// For format details, see https://aka.ms/devcontainer.json. For config options, see the
// README at: https://github.com/devcontainers/templates/tree/main/src/docker-existing-dockerfile
{
	"name": "Employee Attrition Batch Dev Environment",
	"build": {
		// Sets the run context to the project root (one level up from .devcontainer folder).
		"context": "..",
		// Use the Dockerfile.batch for batch predictions.
		"dockerfile": "../Dockerfile.batch"
	},

	// Features to add to the dev container. More info: https://containers.dev/features.
	"features": {
		// Add Python development tools (matching Dockerfile.batch base image).
		"ghcr.io/devcontainers/features/python:1": {
			"version": "3.9"
		},
		// Add Git for version control.
		"ghcr.io/devcontainers/features/git:1": {}
	},

	// No ports need forwarding for batch processing (runs offline), but included for potential debugging.
	"forwardPorts": [],

	// Run commands after the container is created to set up the environment.
	"postCreateCommand": "uv sync && echo 'Batch environment setup complete!'",

	// Configure tool-specific properties for VS Code.
	"customizations": {
		"vscode": {
			// Extensions to install for development.
			"extensions": [
				"ms-python.python", // Python extension for IntelliSense, linting, debugging
				"ms-python.vscode-pylance", // Enhanced Python language support
				"ms-python.black-formatter", // Black formatter for Python code
				"ms-python.isort", // isort for sorting imports
				"ms-toolsai.jupyter", // Jupyter support for notebook-based data exploration
				"GitHub.copilot" // GitHub Copilot for code suggestions
			],
			// VS Code settings for Python development.
			"settings": {
				"python.defaultInterpreterPath": "/usr/local/bin/python",
				"python.linting.enabled": true,
				"python.linting.pylintEnabled": true,
				"python.formatting.provider": "black",
				"editor.formatOnSave": true,
				"editor.defaultFormatter": "ms-python.black-formatter",
				"python.sortImports": true,
				"terminal.integrated.defaultProfile.linux": "bash"
			}
		}
	},

	// Connect as a non-root user for better security.
	"remoteUser": "vscode"
}
```

---

### Explanation of the Configuration

1. **Name**:
   - Set to `"Employee Attrition Batch Dev Environment"` to clearly indicate its purpose for batch processing.

2. **Build**:
   - `"context": ".."`: Uses the project root as the build context, consistent with the project structure.
   - `"dockerfile": "../Dockerfile.batch"`: Specifies the `Dockerfile.batch` for building the container, which is designed for running `predict_model.py`.

3. **Features**:
   - `"ghcr.io/devcontainers/features/python:1"`: Installs Python 3.9 to match the base image in `Dockerfile.batch` (typically `python:3.9-slim`).
   - `"ghcr.io/devcontainers/features/git:1"`: Adds Git for version control, aligning with the project’s use of Git and GitHub Actions.

4. **ForwardPorts**:
   - Left as an empty array (`[]`) since batch processing doesn’t require exposing ports (it runs offline and writes to files). However, it’s included as a placeholder in case you want to debug or add a local server later.

5. **PostCreateCommand**:
   - `"uv sync"`: Installs project dependencies using `uv`, as specified in `pyproject.toml` and `uv.lock`.
   - Adds a confirmation message to verify setup completion.

6. **Customizations**:
   - **VS Code Extensions**:
     - Included `ms-python.python`, `ms-python.vscode-pylance`, `ms-python.black-formatter`, and `ms-python.isort` for consistent Python development, matching the API container’s setup.
     - Added `ms-toolsai.jupyter` for notebook-based data exploration or testing (e.g., `notebooks/` in the project structure).
     - Included `GitHub.copilot` for coding assistance.
   - **VS Code Settings**:
     - Configured the Python interpreter, linting (Pylint), formatting (Black), and import sorting (isort) to align with the project’s dev-dependencies.
     - Set `bash` as the default terminal for Linux container compatibility.

7. **RemoteUser**:
   - Set to `vscode` for a non-root user, enhancing security during development.

---

### Differences from `devcontainer.json` for `Dockerfile.api`

- **Dockerfile**: Uses `Dockerfile.batch` instead of `Dockerfile.api`.
- **ForwardPorts**: No ports are forwarded (unlike port 8000 for FastAPI), as batch processing is file-based.
- **Extensions**: Added `ms-toolsai.jupyter` for notebook support, which is more relevant for batch data exploration than API development.
- **Purpose**: Tailored for offline batch prediction (`predict_model.py`) rather than real-time API serving (`app.py`).

---

### How to Use the `devcontainer-batch.json`

1. **Save the File**:
   - Place the file as `.devcontainer/devcontainer-batch.json` in your project directory.

2. **Open in VS Code**:
   - Open the project folder in VS Code.
   - Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on macOS) to open the command palette.
   - Select **Dev Containers: Reopen Folder in Container** and choose `devcontainer-batch.json` from the list (VS Code will detect multiple dev container configurations).
   - VS Code will build the container using `Dockerfile.batch` and apply the settings.

3. **Verify Setup**:
   - Open a terminal in VS Code.
   - Run `python --version` to confirm Python 3.9 is installed.
   - Run `uv --version` to confirm `uv` is available.
   - Test the batch script: `python src/models/predict_model.py` (ensure `data/interim/new_employees.csv` exists or create a sample file).
   - Check `data/predictions/predictions.csv` for output.

4. **Switch Between Containers**:
   - Use the command palette to reopen the folder with either `devcontainer.json` (for API) or `devcontainer-batch.json` (for batch) as needed.

---

### Additional Considerations

- **Dockerfile.batch Compatibility**:
  - Ensure `Dockerfile.batch` installs `uv` and syncs dependencies (it likely mirrors `Dockerfile.api` in this regard). If it differs (e.g., different base image), adjust the `"version"` in the `python` feature accordingly.

- **Data Volume**:
  - If you need to work with local data files (e.g., `data/interim/`), you might add a `mounts` section to map the `data/` directory:
    ```jsonc
    "mounts": [
        "source=${localWorkspaceFolder}/data,target=/app/data,type=bind"
    ]
    ```
    This requires updating the `Dockerfile.batch` `WORKDIR` to `/app` if it differs.

- **Debugging**:
  - If you need to debug the batch process, consider adding a port for a local server or Jupyter notebook (e.g., forward port 8888 and install Jupyter via `postCreateCommand`).

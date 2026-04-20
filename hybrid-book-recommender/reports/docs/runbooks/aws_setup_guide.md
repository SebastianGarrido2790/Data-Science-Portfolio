# AWS EC2 Setup Guide

In case you do not have the AWS CLI installed, we will set up the infrastructure manually using the AWS Management Console. This follows the "ClickOps" approach, which is fine for a single portfolio instance.

## 1. Login
1.  Go to [AWS Console](https://console.aws.amazon.com/).
2.  Log in to your account.
3.  Ensure you are in a region close to you (e.g., `sa-east-1` Sao Paulo).

## 2. Launch Instance
1.  Search for **EC2** in the top search bar and click it.
2.  Click the orange **Launch Instance** button.
3.  **Name:** `hybrid-book-recommender`

### Application and OS Images (AMI)
*   Select **Ubuntu**.
*   Verified Selection: **Ubuntu Server 24.04 LTS (HVM)** (Free Tier eligible).

### Instance Type
*   Select **t3.micro** or **t2.micro** (Free Tier eligible).

### Key Pair (Login)
*   Click **Create new key pair**.
*   **Key pair name:** `hybrid-recommender-key`
*   **Key pair type:** `RSA`
*   **Private key file format:** `.pem` (for OpenSSH / Linux / Mac / Windows PowerShell)
*   Click **Create key pair**.
*   **IMPORTANT:** A file named `hybrid-recommender-key.pem` will download. **Save this file safely!** (e.g., in your `C:\Users\sebas\.ssh\` folder or plain Downloads).

### Network Settings
*   Click **Edit** (top right of this box).
*   **Auto-assign Public IP:** Enable.
*   **Firewall (Security Groups):** Create security group.
*   **Rule 1 (SSH):** Type: SSH, Port: 22, Source: **My IP** (Preferable) or Anywhere (0.0.0.0/0).
*   **Rule 2 (App):** Click **Add security group rule**.
    *   **Type:** Custom TCP
    *   **Port range:** `7860`
    *   **Source:** Anywhere `0.0.0.0/0` (This allows the public to see your Gradio app).

### Configure Storage
*   Change `8 GiB` to **15 GiB** (Free tier allows up to 30GB, and Docker images can get large).

## 3. Launch
1.  Click **Launch Instance**.
2.  Click the **Instance ID** (e.g., `i-012345...`) to view your running instance.
3.  Wait for **Instance state** to turn `Running`.

## 4. Required Information
Once running, find the **Public IPv4 address** in the details pane.

**Please reply with:**
1.  The **Public IPv4 Address** (e.g., `54.123.45.67`).
2.  The full path to where you saved the `.pem` key file (e.g., `C:\Users\...\Downloads\hybrid-recommender-key.pem`).

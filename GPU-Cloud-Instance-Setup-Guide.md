# Setting Up a GPU Cloud Instance for LLM Serving (Outside Google Colab)

> **Goal:** Run your fine-tuned Qwen3 Pizza Palace model on a real GPU in the cloud, accessible from your Spring Boot app.
> **Audience:** Beginners who have only used Google Colab so far
> **Prerequisites:** Your `pizza_merged_16bit` folder or `.gguf` file from the fine-tune notebook

---

## Table of Contents

1. [Why You Need a GPU Cloud Instance](#1-why-you-need-a-gpu-cloud-instance)
2. [Which GPU Do I Need?](#2-which-gpu-do-i-need)
3. [Cloud Provider Options](#3-cloud-provider-options)
4. [Option A: Google Cloud Platform (GCP) — L4 GPU](#4-option-a-google-cloud-platform-gcp--l4-gpu)
5. [Option B: AWS — g5.xlarge (A10G)](#5-option-b-aws--g5xlarge-a10g)
6. [Option C: RunPod / Vast.ai — Quick GPU Rental](#6-option-c-runpod--vastai--quick-gpu-rental)
7. [Common Setup After Creating the VM](#7-common-setup-after-creating-the-vm)
8. [Deploying the Pizza Palace Model](#8-deploying-the-pizza-palace-model)
9. [Connecting Spring AI to Your GPU Server](#9-connecting-spring-ai-to-your-gpu-server)
10. [Security Checklist](#10-security-checklist)
11. [Cost Management Tips](#11-cost-management-tips)
12. [Troubleshooting](#12-troubleshooting)
13. [Quick Decision Matrix](#13-quick-decision-matrix)

---

## 1. Why You Need a GPU Cloud Instance

Google Colab is great for **training**, but it shuts down when you close the browser. To serve real customers, you need a machine that:

- Stays on 24/7
- Has a public IP address
- Can run vLLM, Ollama, or similar servers
- Is reachable from your Spring Boot backend

```
Customer ──► Your Spring Boot API ──► GPU Cloud Instance ──► vLLM/Ollama ──► Model
                (can be anywhere)        (has GPU + public IP)                (responds)
```

---

## 2. Which GPU Do I Need?

| Model Size | VRAM Needed for 16-bit | Suggested GPU | Cloud Cost (approx) |
|-----------|----------------------|--------------|---------------------|
| 4B (Qwen3-4B) | ~8 GB | NVIDIA L4, A10G, RTX 3090/4090 | $0.50–$1.50/hour |
| 7B–8B | ~16 GB | L4, A10G, RTX 4090, A100 40GB | $0.80–$3/hour |
| 13B | ~26 GB | A100 40GB or 80GB, A10G × 2 | $2–$4/hour |
| 70B | ~140 GB | A100 80GB × 2, H100 | $6–$15/hour |

For the **Pizza Palace bot (Qwen3-4B)**, you only need ~8–10 GB VRAM.

---

## 3. Cloud Provider Options

| Provider | Ease of Use | Cost | Best For |
|----------|-------------|------|----------|
| **Google Cloud (GCP)** | Medium | Medium | Learning, credits for students |
| **AWS** | Medium | Medium | Enterprise, existing AWS users |
| **Azure** | Medium | Medium | Microsoft ecosystem |
| **RunPod** | Easy | Low–Medium | Quick experiments |
| **Vast.ai** | Easy | Cheapest | Hobbyists, lowest cost |
| **Lambdalabs** | Easy | Medium | ML-focused, good support |
| **Paperspace** | Easy | Low | Free tier available |
| **JarvisLabs.ai** | Easy | Low | Indian students, INR billing |

We'll cover **GCP**, **AWS**, and **RunPod** because they represent: cloud giant, enterprise, and simple GPU rental.

---

## 4. Option A: Google Cloud Platform (GCP) — L4 GPU

### 4.1 What You Need

- A Google account
- A GCP billing account (credit card required, but you may have $300 free credits)

### 4.2 Step-by-Step (Web Console)

#### Step 1: Create a Project

```
1. Go to https://console.cloud.google.com
2. Click the project selector at the top
3. Click "New Project"
4. Name it: pizza-palace-llm
5. Click Create
```

#### Step 2: Request GPU Quota

GPUs are not enabled by default.

```
1. Go to IAM & Admin → Quotas
2. Search for "NVIDIA L4 GPUs"
3. Select your region (e.g., us-central1)
4. Click "Edit Quota"
5. Request 1 GPU with reason: "Running a fine-tuned LLM for a student/personal project"
6. Wait for approval (usually minutes to hours)
```

#### Step 3: Create the VM

```
1. Go to Compute Engine → VM instances
2. Click "Create Instance"
3. Name: pizza-palace-gpu
4. Region: us-central1 (or nearest to you)
5. Machine type: g2-standard-4 (has 1x NVIDIA L4)
6. Boot disk: 
   - Operating system: Ubuntu 22.04 LTS
   - Size: 80 GB standard persistent disk (or more)
7. GPU: NVIDIA L4, count: 1
8. Firewall: Allow HTTP and HTTPS traffic
9. Click Create
```

#### Step 4: Open SSH in Browser

```
1. In the VM instances list, click "SSH" next to your instance
2. A browser-based terminal opens
```

#### Step 5: Install NVIDIA Drivers and CUDA

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install NVIDIA driver
sudo apt install -y ubuntu-drivers-common
sudo ubuntu-drivers autoinstall

# Reboot
sudo reboot
```

After reboot, reconnect and verify:

```bash
nvidia-smi
```

You should see the L4 GPU.

### 4.3 GCP CLI Alternative (Faster for Repeats)

```bash
# Install gcloud CLI first: https://cloud.google.com/sdk/docs/install

gcloud auth login
gcloud config set project pizza-palace-llm

# Create instance
gcloud compute instances create pizza-palace-gpu \
  --zone=us-central1-a \
  --machine-type=g2-standard-4 \
  --accelerator=type=nvidia-l4,count=1 \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud \
  --boot-disk-size=80GB \
  --maintenance-policy=TERMINATE \
  --restart-on-failure
```

---

## 5. Option B: AWS — g5.xlarge (A10G)

### 5.1 What You Need

- AWS account
- Credit card for billing

### 5.2 Step-by-Step (AWS Console)

#### Step 1: Request Spot/On-Demand GPU Quota

```
1. Go to AWS Console → Service Quotas
2. Search for "g5.xlarge" or "Amazon EC2 P/G instances"
3. Request a quota increase for your region
4. Reason: "Running a fine-tuned LLM inference server"
```

#### Step 2: Launch EC2 Instance

```
1. EC2 → Instances → Launch Instances
2. Name: pizza-palace-gpu
3. AMI: Deep Learning AMI GPU PyTorch 2.3 (Ubuntu 22.04)
   (This already has NVIDIA drivers, CUDA, Python installed!)
4. Instance type: g5.xlarge
5. Key pair: Create new (download the .pem file!)
6. Network settings: Allow SSH (port 22), HTTP (80), HTTPS (443)
7. Storage: 80 GB gp3
8. Launch instance
```

#### Step 3: Connect via SSH

```bash
# From your local machine
cd /path/to/your/key
chmod 400 your-key.pem

ssh -i your-key.pem ubuntu@YOUR_EC2_PUBLIC_IP
```

#### Step 4: Verify GPU

```bash
nvidia-smi
```

With the Deep Learning AMI, CUDA is already installed.

---

## 6. Option C: RunPod / Vast.ai — Quick GPU Rental

### 6.1 RunPod (Great for Hourly Experiments)

```
1. Sign up at https://www.runpod.io
2. Add payment method
3. Go to "GPU Pods" → "Deploy"
4. Select template: "PyTorch" or "vLLM"
5. Choose GPU: RTX 3090, RTX 4090, A5000, A6000, etc.
6. Set cloud storage: 20 GB minimum
7. Click "Deploy"
8. Wait 1–2 minutes for pod to start
9. Click "Connect" → "Connect to JupyterLab" or "SSH"
```

RunPod gives you a JupyterLab interface or SSH access. Much simpler than GCP/AWS.

### 6.2 Vast.ai (Usually Cheapest)

```
1. Sign up at https://vast.ai
2. Go to "Templates"
3. Search for "vLLM" or "PyTorch"
4. Select a machine with RTX 3090/4090/A5000
5. Click "Rent"
6. Copy the SSH command shown
7. SSH into the machine
```

Example SSH command from Vast.ai:

```bash
ssh -p 12345 root@ssh5.vast.ai -L 8000:localhost:8000
```

The `-L 8000:localhost:8000` part forwards the vLLM port to your local machine so you can test `http://localhost:8000`.

---

## 7. Common Setup After Creating the VM

No matter which provider you chose, the next steps are similar.

### 7.1 Install Python and pip

```bash
sudo apt update
sudo apt install -y python3-pip python3-venv git
```

### 7.2 Create a Virtual Environment

```bash
python3 -m venv vllm-env
source vllm-env/bin/activate
```

### 7.3 Install vLLM

```bash
pip install vllm
```

If you get CUDA errors, check CUDA version:

```bash
nvcc --version
nvidia-smi
```

### 7.4 Upload Your Model

#### From Your Local Machine (scp)

```bash
# If using GCP/AWS/Vast via SSH
scp -i your-key.pem -r pizza_merged_16bit ubuntu@YOUR_SERVER_IP:/home/ubuntu/

# If RunPod, use their file uploader in the web UI
```

#### From Google Drive

```bash
# On the GPU server
pip install gdown

# Download a shared Google Drive zip file
gdown --id YOUR_FILE_ID
unzip pizza_merged_16bit.zip
```

#### From HuggingFace Hub

If you pushed the merged model to HuggingFace:

```bash
pip install huggingface-hub
huggingface-cli download your-username/pizza-qwen3 --local-dir ./pizza_merged_16bit
```

---

## 8. Deploying the Pizza Palace Model

### 8.1 Start vLLM

```bash
source vllm-env/bin/activate

python -m vllm.entrypoints.openai.api_server \
  --model /home/ubuntu/pizza_merged_16bit \
  --served-model-name pizza-palace \
  --max-model-len 1024 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.9 \
  --port 8000 \
  --host 0.0.0.0
```

> **Security warning:** `--host 0.0.0.0` makes it reachable from the internet. Only do this if the port is firewalled or behind authentication. In production, use a reverse proxy with SSL.

### 8.2 Test vLLM

Open another terminal (or run locally with port forwarding):

```bash
curl http://YOUR_SERVER_IP:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "pizza-palace",
    "messages": [
      {"role":"system","content":"You are a Pizza Palace agent. Menu: Margherita ₹210, Cola ₹60. Use only this menu."},
      {"role":"user","content":"what veg pizzas?"}
    ],
    "temperature": 0.4,
    "max_tokens": 128
  }'
```

### 8.3 Run Ollama Instead (Simpler, CPU/GPU)

If you exported the GGUF file and just want Ollama:

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Copy GGUF + Modelfile
mkdir -p ~/pizza-palace
mv unsloth.Q4_K_M.gguf ~/pizza-palace/
cat > ~/pizza-palace/Modelfile << 'EOF'
FROM ./unsloth.Q4_K_M.gguf

SYSTEM "You are a warm, upbeat Pizza Palace agent. Menu: Margherita ₹210, Cola ₹60. Use only this menu."

PARAMETER temperature 0.4
PARAMETER top_p 0.9
EOF

ollama create pizza-palace -f ~/pizza-palace/Modelfile
ollama serve &
ollama run pizza-palace "what veg pizzas?"
```

---

## 9. Connecting Spring AI to Your GPU Server

### 9.1 Get the Server IP

```
GCP:  External IP shown in Compute Engine → VM instances
AWS:  Public IPv4 address in EC2 → Instances
RunPod: "Connect" tab shows TCP endpoint
Vast.ai: SSH command includes the hostname
```

### 9.2 Spring Boot Configuration

```properties
# For vLLM (OpenAI-compatible)
spring.ai.openai.base-url=http://YOUR_SERVER_IP:8000/v1
spring.ai.openai.api-key=not-needed
spring.ai.openai.chat.options.model=pizza-palace
spring.ai.openai.chat.options.temperature=0.4
spring.ai.openai.chat.options.max-tokens=128
```

For Ollama:

```properties
spring.ai.ollama.base-url=http://YOUR_SERVER_IP:11434
spring.ai.ollama.chat.options.model=pizza-palace
spring.ai.ollama.chat.options.temperature=0.4
```

### 9.3 Make Sure Ports Are Open

| Server | Port | GCP Firewall Rule | AWS Security Group |
|--------|------|-------------------|-------------------|
| vLLM | 8000 | Allow TCP 8000 | Inbound TCP 8000 |
| Ollama | 11434 | Allow TCP 11434 | Inbound TCP 11434 |
| SSH | 22 | Allow TCP 22 | Inbound TCP 22 |

### 9.4 Simple Spring Boot Service

```java
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.stereotype.Service;

@Service
public class CloudPizzaService {

    private final ChatClient chatClient;

    public CloudPizzaService(ChatClient.Builder builder) {
        this.chatClient = builder
            .defaultSystem("You are a warm, upbeat Pizza Palace agent. Use only the menu.")
            .build();
    }

    public String ask(String question) {
        return chatClient.prompt()
            .user(question)
            .call()
            .content();
    }
}
```

---

## 10. Security Checklist

```
⚠️ Do not skip these if your server is on the public internet.

□  Only open ports you need (22, 8000/11434 if required)
□  Use a firewall/Security Group to restrict 8000/11434 to your Spring Boot IP
□  Add an API key or reverse proxy (nginx/Caddy) with basic auth in front of vLLM
□  Use HTTPS in production (Caddy can auto-provision Let's Encrypt certificates)
□  Disable password login; use SSH keys only
□  Keep the OS and packages updated: sudo apt update && sudo apt upgrade -y
□  Stop the instance when not in use to avoid charges
□  Do not share your .pem private key
```

### 10.1 Quick Reverse Proxy with Caddy

```bash
# Install Caddy
sudo apt install -y debian-keyring debian-archive-keyring apt-transport-https
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | sudo gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' | sudo tee /etc/apt/sources.list.d/caddy-stable.list
sudo apt update
sudo apt install caddy
```

```bash
# Create Caddyfile
sudo tee /etc/caddy/Caddyfile << 'EOF'
your-domain.com {
    reverse_proxy localhost:8000
    basicauth {
        # username = admin, password = secret123 (generate with caddy hash-password)
        admin $2a$14$...
    }
}
EOF

sudo systemctl reload caddy
```

---

## 11. Cost Management Tips

| Tip | Why |
|-----|-----|
| **Stop instance when not in use** | You are charged per hour the VM is running |
| **Use preemptible/spot VMs** | 60–70% cheaper, but can be interrupted (OK for experiments) |
| **Start small** | L4/A10G is enough for 4B models; don't rent A100 unless needed |
| **Use RunPod/Vast.ai for testing** | Hourly billing, easy to destroy when done |
| **Set billing alerts** | GCP/AWS can email you if you exceed a budget |
| **Delete disks after use** | Storage also costs money |

### 11.1 Estimated Monthly Cost (if left running 24/7)

| GPU | Cost/Hour | 24/7 Monthly |
|-----|-----------|--------------|
| GCP L4 (g2-standard-4) | ~$0.80 | ~$580 |
| AWS g5.xlarge (A10G) | ~$1.00 | ~$730 |
| RunPod RTX 4090 | ~$0.50 | ~$360 |
| Vast.ai RTX 3090 | ~$0.30 | ~$220 |

For development, **turn the instance off when not using it** to reduce costs by 90%+.

---

## 12. Troubleshooting

| Problem | Likely Cause | Fix |
|---------|-------------|-----|
| `nvidia-smi` not found | Driver not installed | Reinstall NVIDIA driver or use Deep Learning AMI |
| `CUDA out of memory` | Model too big for GPU | Use smaller model, 4-bit GGUF, or larger GPU |
| `vLLM` says model not found | Wrong path | Verify `ls /home/ubuntu/pizza_merged_16bit` |
| `curl` times out | Firewall blocking port | Open port in cloud provider firewall |
| `Connection refused` | vLLM not running | Check `ps aux \| grep vllm` |
| `Permission denied (publickey)` | Wrong SSH key or permissions | `chmod 400 key.pem`, use correct user |
| High cost | Instance left running | Stop or delete the VM |

---

## 13. Quick Decision Matrix

```
I want to...

Learn cloud basics and may get student credits
    └── Use GCP L4 or AWS g5.xlarge

Get a GPU in 2 minutes and pay hourly
    └── Use RunPod or Vast.ai

Run production for many users
    └── Use GCP/AWS with vLLM + load balancer

Run local-ish on my own machine
    └── Buy or build a PC with RTX 4090/3090 and install Ubuntu

Spend the least money
    └── Use Vast.ai with RTX 3090, shut down when done
```

---

## 14. Summary

> A **GPU cloud instance** is just a Linux server with an NVIDIA GPU. After you create it, the steps are the same everywhere: install drivers, install vLLM/Ollama, upload your model, start the server, and point Spring AI to the server's IP. Start with RunPod or Vast.ai for experiments, then move to GCP/AWS when you need reliability for production.

---

*Guide created for deploying the Qwen3 Pizza Palace fine-tuned model outside Google Colab.*

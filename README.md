<p align="center">
    <h1 align="center">🎙️ Voice Agent Course 🎙️</h1>
    <h3 align="center">Part 2: From WhatsApp to Voice - Building Real-Time Conversational AI</h3>
</p>

> ⚠️ **Work in Progress** - This course is currently under development. Right now, you can test the basic voice conversation demo in `scripts/voice_agent_demo.py`. The full real estate agent features are coming soon!

## Table of Contents

- [Table of Contents](#table-of-contents)
- [Course Overview](#course-overview)
- [Who is this course for?](#who-is-this-course-for)
- [What you'll get out of this course](#what-youll-get-out-of-this-course)
- [Getting started](#getting-started)
- [How much is this going to cost me?](#how-much-is-this-going-to-cost-me)
- [Contributors](#contributors)
- [License](#license)

## Course Overview

Imagine calling a **real estate agent** and instantly getting the information you need. 🏠

No hold times, no missed calls. In this course, we'll guide you step by step to build a fully functional **Real Estate Voice Agent**.

This smart assistant will be **accessible by phone**, ready to **answer property questions**, **schedule viewings**, and make the **buying or renting journey smoother** than ever!

By the end of this course, you'll have built your own voice agent, capable of:

* Answering calls and understanding customer inquiries 📞
* Providing detailed property information instantly 🏘️
* Scheduling viewings and appointments 📅
* Accessing property databases and CRM systems 💾
* Handling multiple calls simultaneously ⚡
* Operating 24/7 without breaks 🌙

>Transform customer service with AI that never sleeps!

Excited? Let's get started! 

**Try the current demo:**
```bash
uv sync
uv run python scripts/voice_agent_demo.py
```

> 🚧 **Current Status**: Basic voice conversation works! The real estate features (property database, scheduling, phone integration) are in development.

---

<table style="border-collapse: collapse; border: none;">
  <tr style="border: none;">
    <td width="20%" style="border: none;">
      <a href="https://theneuralmaze.substack.com/" aria-label="The Neural Maze">
        <img src="https://avatars.githubusercontent.com/u/151655127?s=400&u=2fff53e8c195ac155e5c8ee65c6ba683a72e655f&v=4" alt="The Neural Maze Logo" width="150"/>
      </a>
    </td>
    <td width="80%" style="border: none;">
      <div>
        <h2>📬 Stay Updated</h2>
        <p><b><a href="https://theneuralmaze.substack.com/">Join The Neural Maze</a></b> and learn to build AI Systems that actually work, from principles to production. Every Wednesday, directly to your inbox. Don't miss out!</p>
      </div>
    </td>
  </tr>
</table>

<p align="center">
  <a href="https://theneuralmaze.substack.com/">
    <img src="https://img.shields.io/static/v1?label&logo=substack&message=Subscribe%20Now&style=for-the-badge&color=black&scale=2" alt="Subscribe Now" height="40">
  </a>
</p>

<table style="border-collapse: collapse; border: none;">
  <tr style="border: none;">
    <td width="20%" style="border: none;">
      <a href="https://www.youtube.com/@jesuscopado-en" aria-label="Jesus Copado YouTube Channel">
        <img src="img/JesusCopado_YouTube_Channel_Callout.png" alt="Jesus Copado YouTube Channel" width="150"/>
      </a>
    </td>
    <td width="80%" style="border: none;">
      <div>
        <h2>🎥 Watch More Content</h2>
        <p><b><a href="https://www.youtube.com/@jesuscopado-en">Join Jesús Copado on YouTube</a></b> to explore how to build real AI projects—from voice agents to creative tools. Weekly videos with code, demos, and ideas that push what's possible with AI. Don't miss the next drop!</p>
      </div>
    </td>
  </tr>
</table>

<p align="center">
  <a href="https://www.youtube.com/@jesuscopado-en">
    <img src="https://img.shields.io/static/v1?label&logo=youtube&message=Subscribe%20Now&style=for-the-badge&color=FF0000&scale=2" alt="Subscribe Now" height="40">
  </a>
</p>

---

## Who is this course for?

This course is for Software Engineers, ML Engineers, and AI Engineers who want to level up by building real-time voice AI applications. It's not just a basic tutorial—it's a deep dive into making production-ready voice agents.

## What you'll get out of this course

* Build a fully functional real estate voice agent that handles phone calls
* Learn to integrate with property databases and CRM systems
* Implement appointment scheduling and customer management
* Set up phone integration for real-world deployment
* Build intelligent conversation flows for sales and support
* Deploy production-ready voice AI on cloud infrastructure
* Handle multiple simultaneous calls with smart routing
* Create voice agents for any business vertical

## Getting started

Before you begin, you need:

1. **Install Ollama** and pull a model:
   ```bash
   ollama serve
   ollama pull qwen3:1.7b
   ```

2. **Install dependencies**:
   ```bash
   uv sync
   ```

3. **Test your setup**:
   ```bash
   uv run python scripts/voice_agent_demo.py
   ```

> Make sure your microphone and speakers are working! This demo shows the core voice capabilities that we'll extend into a full real estate agent.

## How much is this going to cost me?

The awesome thing about this project is **you can run it on your own computer for free!**

Everything runs locally - no API costs, no cloud bills. Perfect for development and testing your real estate voice agent.

When we deploy to production with phone integration, we'll show you cost-effective cloud options that scale with your business.

## Contributors

<table>
  <tr>
    <td align="center"><img src="https://github.com/MichaelisTrofficus.png" width="100" style="border-radius:50%;"/></td>
    <td>
      <strong>Miguel Otero Pedrido | Senior ML / AI Engineer </strong><br />
      <i>Founder of The Neural Maze. Rick and Morty fan.</i><br /><br />
      <a href="https://www.linkedin.com/in/migueloteropedrido/">LinkedIn</a><br />
      <a href="https://www.youtube.com/@TheNeuralMaze">YouTube</a><br />
      <a href="https://theneuralmaze.substack.com/">The Neural Maze Newsletter</a>
    </td>
  </tr>
  <tr>
    <td align="center"><img src="https://github.com/jesuscopado.png" width="100" style="border-radius:50%;"/></td>
    <td>
      <strong>Jesús Copado | Senior ML / AI Engineer </strong><br />
      <i>Equal parts cinema fan and AI enthusiast.</i><br /><br />
      <a href="https://www.youtube.com/@jesuscopado-en">YouTube</a><br />
      <a href="https://www.linkedin.com/in/copadojesus/">LinkedIn</a><br />
    </td>
  </tr>
</table>

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<table style="border-collapse: collapse; border: none;">
  <tr style="border: none;">
    <td width="20%" style="border: none;">
      <a href="https://theneuralmaze.substack.com/" aria-label="The Neural Maze">
        <img src="https://avatars.githubusercontent.com/u/151655127?s=400&u=2fff53e8c195ac155e5c8ee65c6ba683a72e655f&v=4" alt="The Neural Maze Logo" width="150"/>
      </a>
    </td>
    <td width="80%" style="border: none;">
      <div>
        <h2>📬 Stay Updated</h2>
        <p><b><a href="https://theneuralmaze.substack.com/">Join The Neural Maze</a></b> and learn to build AI Systems that actually work, from principles to production. Every Wednesday, directly to your inbox. Don't miss out!</p>
      </div>
    </td>
  </tr>
</table>

<p align="center">
  <a href="https://theneuralmaze.substack.com/">
    <img src="https://img.shields.io/static/v1?label&logo=substack&message=Subscribe%20Now&style=for-the-badge&color=black&scale=2" alt="Subscribe Now" height="40">
  </a>
</p>

<table style="border-collapse: collapse; border: none;">
  <tr style="border: none;">
    <td width="20%" style="border: none;">
      <a href="https://www.youtube.com/@jesuscopado-en" aria-label="Jesus Copado YouTube Channel">
        <img src="img/JesusCopado_YouTube_Channel_Callout.png" alt="Jesus Copado YouTube Channel" width="150"/>
      </a>
    </td>
    <td width="80%" style="border: none;">
      <div>
        <h2>🎥 Watch More Content</h2>
        <p><b><a href="https://www.youtube.com/@jesuscopado-en">Join Jesús Copado on YouTube</a></b> to explore how to build real AI projects—from voice agents to creative tools. Weekly videos with code, demos, and ideas that push what's possible with AI. Don't miss the next drop!</p>
      </div>
    </td>
  </tr>
</table>

<p align="center">
  <a href="https://www.youtube.com/@jesuscopado-en">
    <img src="https://img.shields.io/static/v1?label&logo=youtube&message=Subscribe%20Now&style=for-the-badge&color=FF0000&scale=2" alt="Subscribe Now" height="40">
  </a>
</p>
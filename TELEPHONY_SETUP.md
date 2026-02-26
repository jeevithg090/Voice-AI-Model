# Telephony Setup Guide (Twilio Media Streams)

This guide explains how to connect your running Voice AI Agent to a real phone number using **Twilio Media Streams**.

## Prerequisites
1.  **Twilio Account** + Phone Number.
2.  **ngrok** installed locally (to expose your localhost to the internet).
3.  **Python dependencies**: `flask-sock`, `pyngrok` (Already installed).

## Step 1: Expose your Server
Twilio needs to reach your local server.
```bash
# Expose port 8080
ngrok http 8080
```
Copy the Forwarding URL (e.g., `https://a1b2-c3d4.ngrok-free.app`).

## Step 2: Create TwiML Bin
1.  Go to [Twilio Console > Developer Tools > TwiML Bins](https://console.twilio.com/us1/develop/twi-ml/bins).
2.  Create a new Bin called "AI Stream".
3.  Paste this XML code:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say>Connecting you to the AI Assistant.</Say>
    <Connect>
        <Stream url="wss://YOUR-NGROK-URL.ngrok-free.app/media-stream">
            <Parameter name="language" value="hi" />
        </Stream>
    </Connect>
</Response>
```
**IMPORTANT**: Replace `YOUR-NGROK-URL` with the ID from Step 1.
Notice the protocol is `wss://` (Secure WebSocket).
`language` is optional; if omitted the server auto-detects.

## Step 3: Configure Phone Number
1.  Go to **Phone Numbers > Manage > Active Numbers**.
2.  Click your number.
3.  Scroll to **Voice & Fax**.
4.  **A Call Comes In**: Select "TwiML Bin".
5.  Select "AI Stream" (the bin you just created).
6.  Save.

## Step 4: Run the Server
```bash
python src/realtime/signaling_server.py
```

## Step 5: Test
Call your Twilio number.
1.  You should hear "Connecting you to the AI Assistant."
2.  Speak. The server logs should show:
    - `📞 Twilio Connection Incoming...`
    - `📞 Call Started (SID: ...)`
    - `🗣 Speech started`
3.  The agent will reply to you over the phone!

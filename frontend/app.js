document.addEventListener("DOMContentLoaded", () => {
    const wsUrl = "ws://localhost:8767";
    const statusEl = document.getElementById("connection-status");
    const marketsContainer = document.getElementById("markets-container");
    const agentsContainer = document.getElementById("agents-container");
    const tradeFeedContainer = document.getElementById("trade-feed-container");

    // --- NEW: Form elements ---
    const spawnForm = document.getElementById("spawn-form");
    const spawnStatus = document.getElementById("spawn-status");

    // Declare socket in the outer scope so it can be accessed by handlers
    let socket;

    function connect() {
        // Assign to the outer 'socket' variable
        socket = new WebSocket(wsUrl);

        socket.onopen = () => {
            console.log("Connected to exchange server.");
            statusEl.textContent = "CONNECTED";
            statusEl.className = "status-connected";
            
            // Register as a spectator to get the full state
            socket.send(JSON.stringify({
                action: "register_spectator"
            }));
        };

        socket.onmessage = (event) => {
            const data = JSON.parse(event.data);
            console.log("Received:", data); // Good for debugging

            // Route messages to the correct handler
            switch (data.type) {
                case "price_update":
                    renderMarket(data);
                    break;
                case "account_update":
                    renderAgent(data);
                    break;
                case "trade_executed":
                    addTradeToFeed(data);
                    break;
                case "new_market":
                    addMarketToFeed(data);
                    break;
                case "market_resolved":
                    resolveMarket(data);
                    break;
                // You can add a 'case "error":' here to display errors
                case "error":
                    console.error("Server Error:", data.message);
                    spawnStatus.textContent = `Server Error: ${data.message}`;
                    spawnStatus.style.color = "red";
                    break;
            }
        };

        socket.onclose = () => {
            console.log("Disconnected. Reconnecting in 3s...");
            statusEl.textContent = "DISCONNECTED";
            statusEl.className = "status-disconnected";
            socket = null; // Clear the socket
            setTimeout(connect, 3000); // Try to reconnect
        };

        socket.onerror = (err) => {
            console.error("WebSocket Error:", err);
            statusEl.textContent = "ERROR";
            statusEl.className = "status-disconnected";
            socket = null; // Clear the socket
        };
    }

    function renderMarket(data) {
        const { market, price_yes, price_no, q_yes, q_no } = data;
        const marketId = `market-${market.replace(/[^a-zA-Z0-9]/g, '_')}`; // Create safe ID
        
        let marketEl = document.getElementById(marketId);
        if (!marketEl) {
            // Market doesn't exist, create it
            marketEl = document.createElement("div");
            marketEl.id = marketId;
            marketEl.className = "market";
            marketsContainer.prepend(marketEl); // Add new markets to the top
        }
        
        // Update/set the content
        marketEl.innerHTML = `
            <div class="market-name">${market}</div>
            <div class="market-prices">
                <span class="price-yes">YES: $${price_yes.toFixed(4)}</span>
                <span class="price-no">NO: $${price_no.toFixed(4)}</span>
            </div>
            <div class="market-pool">
                <span>q_yes: ${q_yes.toFixed(2)}</span>
                <span>q_no: ${q_no.toFixed(2)}</span>
            </div>
        `;
    }

    function renderAgent(data) {
        const agent_id = data.agent_id; // Get agent_id from our modified message
        if (!agent_id) return; // Sometimes we get updates without an ID, ignore
        
        const balances = data.balances;
        
        const agentId = `agent-${agent_id.replace(/[^a-zA-Z0-9]/g, '_')}`;
        
        let agentEl = document.getElementById(agentId);
        if (!agentEl) {
            agentEl = document.createElement("div");
            agentEl.id = agentId;
            agentEl.className = "agent";
            agentsContainer.prepend(agentEl);
        }
        
        // Filter out zero balances and spectator accounts
        const holdings = Object.entries(balances)
            .filter(([asset, amount]) => amount !== 0)
            .sort((a, b) => a[0].localeCompare(b[0])); // Sort assets alphabetically

        if (holdings.length === 0 || agent_id.startsWith('spectator')) {
            if (agentEl) agentEl.remove(); // Clean up empty/spectator agents
            return;
        }

        const holdingsHtml = holdings.map(([asset, amount]) => `
            <li>
                <span class="holding-asset">${asset}</span>
                <span class="holding-amount">${amount.toFixed(3)}</span>
            </li>
        `).join('');

        agentEl.innerHTML = `
            <div class="agent-name">${agent_id}</div>
            <div class="agent-holdings">
                <ul>${holdingsHtml}</ul>
            </div>
        `;
    }

    function addTradeToFeed(data) {
        const { agent, market, quantity, cost } = data;
        const p = document.createElement("p");
        
        if (quantity > 0) {
            p.className = "trade-buy";
            p.innerHTML = `<strong>${agent}</strong> bought ${quantity.toFixed(2)} ${market} for $${cost.toFixed(2)}`;
        } else {
            p.className = "trade-sell";
            p.innerHTML = `<strong>${agent}</strong> sold ${Math.abs(quantity).toFixed(2)} ${market} for $${(-cost).toFixed(2)}`;
        }
        tradeFeedContainer.prepend(p);
    }

    function addMarketToFeed(data) {
        const p = document.createElement("p");
        p.innerHTML = `🔥 <strong>NEW MARKET:</strong> ${data.market_name}`;
        tradeFeedContainer.prepend(p);
    }
    
    function resolveMarket(data) {
        const { market_name, outcome } = data;
        // Add to feed
        const p = document.createElement("p");
        p.className = "trade-resolve";
        p.innerHTML = `🏁 <strong>MARKET RESOLVED:</strong> ${market_name} resolved to ${outcome}`;
        tradeFeedContainer.prepend(p);
        
        // Remove market from UI (handles both _YES and _NO assets)
        const marketBaseId = market_name.replace(/[^a-zA-Z0-9]/g, '_');
        const marketIdYes = `market-${marketBaseId}_YES`;
        const marketIdNo = `market-${marketBaseId}_NO`;
        
        const marketElYes = document.getElementById(marketIdYes);
        const marketElNo = document.getElementById(marketIdNo);
        
        if (marketElYes) marketElYes.remove();
        if (marketElNo) marketElNo.remove();
    }

    // --- Event Listener for the Spawn Form ---
    if (spawnForm) {
        spawnForm.addEventListener("submit", (e) => {
            e.preventDefault();
            spawnStatus.textContent = "";

            const agentName = document.getElementById("agent-name").value;
            const agentModel = document.getElementById("agent-model").value;
            const agentStrategy = document.getElementById("agent-strategy").value;

            if (!socket || socket.readyState !== WebSocket.OPEN) {
                spawnStatus.textContent = "Error: Not connected to exchange.";
                spawnStatus.style.color = "red";
                return;
            }

            if (!agentName || !agentModel || !agentStrategy) {
                spawnStatus.textContent = "Error: All fields are required.";
                spawnStatus.style.color = "red";
                return;
            }

            console.log("about to send spawn request")

            // Send the spawn command to the exchange server
            socket.send(JSON.stringify({
                action: "spawn_agent",
                name: agentName,
                model: agentModel,
                strategy: agentStrategy
            }));

            console.log("spawn request sent")

            spawnStatus.textContent = `Spawn command sent for ${agentName}...`;
            spawnStatus.style.color = "green";
            
            // Add to the live feed
            const p = document.createElement("p");
            p.innerHTML = `🚀 <strong>SPAWN COMMAND:</strong> Sent for ${agentName} (Model: ${agentModel})`;
            tradeFeedContainer.prepend(p);
            
            // Clear name for next agent
            document.getElementById("agent-name").value = "";
        });
    }

    // Start the connection
    connect();
});
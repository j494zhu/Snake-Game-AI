const GRID_SIZE = 20;
const CELL_SIZE = 20;
const API_URL = "/predict";

let gameSpeed = 100; // milliseconds per frame
let aiConnected = true;

// Speed slider handler
document.getElementById('speedSlider').addEventListener('input', (e) => {
    gameSpeed = parseInt(e.target.value);
    document.getElementById('speedValue').textContent = gameSpeed + 'ms';
});

class SnakeGame {
    constructor(canvasId, isAI = false) {
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas.getContext('2d');
        this.isAI = isAI;
        this.reset();
    }

    reset() {
        this.snake = [{x: 10, y: 10}, {x: 9, y: 10}, {x: 8, y: 10}];
        this.direction = 0; // 0:Right, 1:Down, 2:Left, 3:Up (matches Rust MOVES)
        this.food = this.placeFood();
        this.score = 0;
        this.done = false;
        this.stepsSinceFood = 0;
    }

    placeFood() {
        let x, y;
        while (true) {
            x = Math.floor(Math.random() * GRID_SIZE);
            y = Math.floor(Math.random() * GRID_SIZE);
            if (!this.snake.some(s => s.x === x && s.y === y)) break;
        }
        return {x, y};
    }

    step(action = 0) {
        if (this.done) return;
        this.stepsSinceFood++;

        // Action: 0:Straight, 1:Right, 2:Left
        if (action === 1) this.direction = (this.direction + 1) % 4;
        else if (action === 2) this.direction = (this.direction + 3) % 4;

        const moves = [[1, 0], [0, 1], [-1, 0], [0, -1]];
        const head = this.snake[0];
        const newHead = { x: head.x + moves[this.direction][0], y: head.y + moves[this.direction][1] };

        // Collision check
        if (newHead.x < 0 || newHead.x >= GRID_SIZE || newHead.y < 0 || newHead.y >= GRID_SIZE ||
            this.snake.some((s, i) => i !== this.snake.length - 1 && s.x === newHead.x && s.y === newHead.y)) {
            this.done = true;
            return;
        }

        this.snake.unshift(newHead);
        if (newHead.x === this.food.x && newHead.y === this.food.y) {
            this.score++;
            this.food = this.placeFood();
            this.stepsSinceFood = 0;
        } else {
            this.snake.pop();
        }

        if (this.stepsSinceFood > 2000) this.done = true;
    }

    // Check if position collides with wall or snake body
    isCollision(x, y) {
        if (x < 0 || x >= GRID_SIZE || y < 0 || y >= GRID_SIZE) return true;
        // Check snake body (excluding tail which will move)
        for (let i = 0; i < this.snake.length - 1; i++) {
            if (this.snake[i].x === x && this.snake[i].y === y) return true;
        }
        return false;
    }

    // Flood fill to count reachable cells (normalized)
    floodFillCount(startX, startY) {
        if (startX < 0 || startX >= GRID_SIZE || startY < 0 || startY >= GRID_SIZE) return 0.0;
        
        const snakeSet = new Set();
        this.snake.forEach(s => snakeSet.add(`${s.x},${s.y}`));
        if (snakeSet.has(`${startX},${startY}`)) return 0.0;
        
        const totalEmpty = GRID_SIZE * GRID_SIZE - this.snake.length;
        if (totalEmpty === 0) return 0.0;
        
        const visited = new Set();
        visited.add(`${startX},${startY}`);
        const queue = [{x: startX, y: startY}];
        let count = 0;
        const moves = [[1, 0], [0, 1], [-1, 0], [0, -1]];
        
        while (queue.length > 0) {
            const {x, y} = queue.shift();
            count++;
            for (const [dx, dy] of moves) {
                const nx = x + dx, ny = y + dy;
                const key = `${nx},${ny}`;
                if (nx >= 0 && nx < GRID_SIZE && ny >= 0 && ny < GRID_SIZE &&
                    !visited.has(key) && !snakeSet.has(key)) {
                    visited.add(key);
                    queue.push({x: nx, y: ny});
                }
            }
        }
        return count / totalEmpty;
    }

    // Get state matching the Rust implementation
    getState() {
        const spatial = new Array(3).fill(0).map(() => new Array(20).fill(0).map(() => new Array(20).fill(0)));
        const aux = new Array(21).fill(0);
        const moves = [[1, 0], [0, 1], [-1, 0], [0, -1]]; // Right, Down, Left, Up

        const hx = this.snake[0].x;
        const hy = this.snake[0].y;

        // Spatial Channels
        spatial[0][hy][hx] = 1.0; // Head
        this.snake.slice(1).forEach((s, i) => {
            const val = 1.0 - (i / (this.snake.length + 1)) * 0.5;
            spatial[1][s.y][s.x] = val; // Body with gradient
        });
        spatial[2][this.food.y][this.food.x] = 1.0; // Food

        // Aux[0-3]: Direction one-hot
        aux[this.direction] = 1.0;

        // Aux[4-12]: Collision detection at depths 1-3 for forward, right, left
        for (let depth = 1; depth <= 3; depth++) {
            const base = 4 + (depth - 1) * 3;
            // Forward
            const [fdx, fdy] = moves[this.direction];
            aux[base] = this.isCollision(hx + fdx * depth, hy + fdy * depth) ? 1.0 : 0.0;
            // Right
            const rightDir = (this.direction + 1) % 4;
            const [rdx, rdy] = moves[rightDir];
            aux[base + 1] = this.isCollision(hx + rdx * depth, hy + rdy * depth) ? 1.0 : 0.0;
            // Left
            const leftDir = (this.direction + 3) % 4;
            const [ldx, ldy] = moves[leftDir];
            aux[base + 2] = this.isCollision(hx + ldx * depth, hy + ldy * depth) ? 1.0 : 0.0;
        }

        // Aux[13-16]: Food direction relative to head
        const fx = this.food.x, fy = this.food.y;
        aux[13] = fy < hy ? 1.0 : 0.0; // Food is up
        aux[14] = fy > hy ? 1.0 : 0.0; // Food is down
        aux[15] = fx < hx ? 1.0 : 0.0; // Food is left
        aux[16] = fx > hx ? 1.0 : 0.0; // Food is right

        // Aux[17]: Snake length normalized
        aux[17] = this.snake.length / 400.0;

        // Aux[18-20]: Flood fill counts for forward, right, left
        const dirs = [this.direction, (this.direction + 1) % 4, (this.direction + 3) % 4];
        for (let i = 0; i < 3; i++) {
            const [dx, dy] = moves[dirs[i]];
            aux[18 + i] = this.floodFillCount(hx + dx, hy + dy);
        }

        return { spatial, aux };
    }

    draw() {
        const ctx = this.ctx;
        
        // Background
        ctx.fillStyle = "#0f172a";
        ctx.fillRect(0, 0, 400, 400);
        
        // Grid lines (subtle)
        ctx.strokeStyle = "#1e293b";
        ctx.lineWidth = 1;
        for (let i = 0; i <= GRID_SIZE; i++) {
            ctx.beginPath();
            ctx.moveTo(i * CELL_SIZE, 0);
            ctx.lineTo(i * CELL_SIZE, 400);
            ctx.stroke();
            ctx.beginPath();
            ctx.moveTo(0, i * CELL_SIZE);
            ctx.lineTo(400, i * CELL_SIZE);
            ctx.stroke();
        }

        // Draw food with glow effect
        ctx.shadowColor = "#fbbf24";
        ctx.shadowBlur = 15;
        ctx.fillStyle = "#fbbf24";
        ctx.beginPath();
        ctx.arc(
            this.food.x * CELL_SIZE + CELL_SIZE / 2,
            this.food.y * CELL_SIZE + CELL_SIZE / 2,
            CELL_SIZE / 2 - 2,
            0, Math.PI * 2
        );
        ctx.fill();
        ctx.shadowBlur = 0;

        // Draw snake
        this.snake.forEach((s, i) => {
            const isHead = i === 0;
            if (isHead) {
                ctx.fillStyle = this.isAI ? "#ef4444" : "#10b981";
                ctx.shadowColor = this.isAI ? "#ef4444" : "#10b981";
                ctx.shadowBlur = 10;
            } else {
                const alpha = 1 - (i / this.snake.length) * 0.5;
                ctx.fillStyle = this.isAI ? `rgba(185, 28, 28, ${alpha})` : `rgba(6, 95, 70, ${alpha})`;
                ctx.shadowBlur = 0;
            }
            
            const padding = isHead ? 1 : 2;
            ctx.beginPath();
            ctx.roundRect(
                s.x * CELL_SIZE + padding,
                s.y * CELL_SIZE + padding,
                CELL_SIZE - padding * 2,
                CELL_SIZE - padding * 2,
                isHead ? 6 : 4
            );
            ctx.fill();
        });
        ctx.shadowBlur = 0;
    }
}

let aiGame, playerGame, mode;

function startGame(m) {
    mode = m;
    document.getElementById('menu').classList.add('hidden');
    document.getElementById('game-area').classList.remove('hidden');

    aiGame = new SnakeGame('aiCanvas', true);
    if (mode === 'versus') {
        playerGame = new SnakeGame('playerCanvas', false);
        document.getElementById('player-side').classList.remove('hidden');
        // Show countdown for versus mode
        showCountdown(() => gameLoop());
    } else {
        gameLoop();
    }
}

function showCountdown(callback) {
    const overlay = document.getElementById('countdown-overlay');
    const numberEl = document.getElementById('countdown-number');
    overlay.classList.remove('hidden');
    
    let count = 3;
    numberEl.textContent = count;
    numberEl.className = 'countdown-number';
    
    // Draw initial state
    aiGame.draw();
    playerGame.draw();
    
    const interval = setInterval(() => {
        count--;
        if (count > 0) {
            numberEl.textContent = count;
            numberEl.className = 'countdown-number';
            // Force re-trigger animation
            void numberEl.offsetWidth;
            numberEl.classList.add('countdown-number');
        } else if (count === 0) {
            numberEl.textContent = 'GO!';
            numberEl.className = 'countdown-number go';
        } else {
            clearInterval(interval);
            overlay.classList.add('hidden');
            callback();
        }
    }, 1000);
}

function updateStatus(connected) {
    aiConnected = connected;
    const dot = document.getElementById('statusDot');
    const text = document.getElementById('statusText');
    if (connected) {
        dot.classList.remove('error');
        text.textContent = 'AI Connected';
    } else {
        dot.classList.add('error');
        text.textContent = 'AI Offline';
    }
}

async function gameLoop() {
    // AI decision
    if (!aiGame.done) {
        const state = aiGame.getState();
        try {
            const resp = await fetch(API_URL, {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(state)
            });
            const result = await resp.json();
            aiGame.step(result.action);
            updateStatus(true);
        } catch (e) {
            console.error("AI Backend Down");
            updateStatus(false);
            aiGame.step(0); // Continue straight if backend is down
        }
    }

    // Player logic
    if (mode === 'versus' && playerGame && !playerGame.done) {
        playerGame.step(0); // Default: go straight
    }

    // Draw
    aiGame.draw();
    document.getElementById('ai-score').innerText = aiGame.score;
    if (playerGame) {
        playerGame.draw();
        document.getElementById('p-score').innerText = playerGame.score;
    }

    // Check game over
    const aiDone = aiGame.done;
    const playerDone = !playerGame || playerGame.done;

    if ((mode === 'demo' && aiDone) || (mode === 'versus' && aiDone && playerDone)) {
        showGameOver();
        return;
    }
    
    setTimeout(gameLoop, gameSpeed);
}

function showGameOver() {
    document.getElementById('ai-final').innerText = aiGame.score;
    
    if (playerGame) {
        document.getElementById('player-final-score').classList.remove('hidden');
        document.getElementById('p-final').innerText = playerGame.score;
        
        if (playerGame.score > aiGame.score) {
            document.getElementById('result-text').innerText = 'You Win!';
            document.getElementById('result-text').style.color = '#10b981';
        } else if (playerGame.score < aiGame.score) {
            document.getElementById('result-text').innerText = 'AI Wins!';
            document.getElementById('result-text').style.color = '#ef4444';
        } else {
            document.getElementById('result-text').innerText = "It's a Tie!";
            document.getElementById('result-text').style.color = '#fbbf24';
        }
    } else {
        document.getElementById('result-text').innerText = 'Game Over';
    }
    
    document.getElementById('game-over-ui').classList.remove('hidden');
}

// Player controls - ABSOLUTE direction (standard snake controls)
// Press a direction key to move in that direction
window.addEventListener('keydown', e => {
    if (!playerGame || playerGame.done) return;
    
    // Prevent page scrolling with arrow keys
    if (['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight', ' '].includes(e.key)) {
        e.preventDefault();
    }
    
    const currentDir = playerGame.direction;
    // Direction mapping: 0:Right, 1:Down, 2:Left, 3:Up
    
    // Go UP (W or Up Arrow) - can't go up if currently going down
    if (e.key === 'w' || e.key === 'W' || e.key === 'ArrowUp') {
        if (currentDir !== 1) playerGame.direction = 3;
    }
    // Go DOWN (S or Down Arrow) - can't go down if currently going up
    if (e.key === 's' || e.key === 'S' || e.key === 'ArrowDown') {
        if (currentDir !== 3) playerGame.direction = 1;
    }
    // Go LEFT (A or Left Arrow) - can't go left if currently going right
    if (e.key === 'a' || e.key === 'A' || e.key === 'ArrowLeft') {
        if (currentDir !== 0) playerGame.direction = 2;
    }
    // Go RIGHT (D or Right Arrow) - can't go right if currently going left
    if (e.key === 'd' || e.key === 'D' || e.key === 'ArrowRight') {
        if (currentDir !== 2) playerGame.direction = 0;
    }
});

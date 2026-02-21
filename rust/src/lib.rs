use pyo3::prelude::*;

use numpy::{PyArray1, PyArray2, PyArray3, PyArray4, IntoPyArray};
use rand::prelude::*;
use rayon::prelude::*;
use std::collections::VecDeque;

const GRID_SIZE: i32 = 20;
const MAX_STEPS_RATIO: i32 = 50; 

const MOVES: [(i32, i32); 4] = [(1, 0), (0, 1), (-1, 0), (0, -1)];


struct SnakeGame {
    snake: VecDeque<(i32, i32)>,
    snake_set: Vec<bool>, 
    food: Option<(i32, i32)>,
    score: i32,
    steps_since_food: i32,
    direction: i32,
    bfs_visited: Vec<bool>,
    bfs_queue: VecDeque<(i32, i32)>,
}

impl SnakeGame {
    fn new() -> Self {
        let mut game = SnakeGame {
            snake: VecDeque::with_capacity(400),
            snake_set: vec![false; (GRID_SIZE * GRID_SIZE) as usize],
            food: None,
            score: 0,
            steps_since_food: 0,
            direction: 0,
            bfs_visited: vec![false; (GRID_SIZE * GRID_SIZE) as usize],
            bfs_queue: VecDeque::with_capacity(400),
        };
        game.reset();
        game
    }

    fn reset(&mut self) {
        self.direction = 0;
        self.score = 0;
        self.steps_since_food = 0;
        let center = GRID_SIZE / 2;
        self.snake.clear();
        self.snake.push_back((center, center));
        self.snake.push_back((center - 1, center));
        self.snake.push_back((center - 2, center));
        self.snake_set.fill(false);
        for &(x, y) in &self.snake {
            self.snake_set[(y * GRID_SIZE + x) as usize] = true;
        }
        self.place_food();
    }

    fn place_food(&mut self) {
        let total_cells = GRID_SIZE * GRID_SIZE;
        let snake_len = self.snake.len() as i32;
        let mut rng = thread_rng();
        if snake_len >= total_cells {
            self.food = None;
            return;
        }
        if snake_len < total_cells / 2 {
            loop {
                let x = rng.gen_range(0..GRID_SIZE);
                let y = rng.gen_range(0..GRID_SIZE);
                if !self.snake_set[(y * GRID_SIZE + x) as usize] {
                    self.food = Some((x, y));
                    return;
                }
            }
        } else {
            let mut empty_cells = Vec::with_capacity((total_cells - snake_len) as usize);
            for y in 0..GRID_SIZE {
                for x in 0..GRID_SIZE {
                    if !self.snake_set[(y * GRID_SIZE + x) as usize] {
                        empty_cells.push((x, y));
                    }
                }
            }
            if let Some(&pos) = empty_cells.choose(&mut rng) {
                self.food = Some(pos);
            } else {
                self.food = None;
            }
        }
    }

    fn flood_fill_count(&mut self, start_x: i32, start_y: i32) -> f32 {
        if start_x < 0 || start_x >= GRID_SIZE || start_y < 0 || start_y >= GRID_SIZE { return 0.0; }
        if self.snake_set[(start_y * GRID_SIZE + start_x) as usize] { return 0.0; }
        self.bfs_visited.copy_from_slice(&self.snake_set);
        self.bfs_queue.clear();
        self.bfs_queue.push_back((start_x, start_y));
        self.bfs_visited[(start_y * GRID_SIZE + start_x) as usize] = true;
        let mut count = 0;
        let total_empty = (GRID_SIZE * GRID_SIZE) as usize - self.snake.len();
        if total_empty == 0 { return 0.0; }
        while let Some((cx, cy)) = self.bfs_queue.pop_front() {
            count += 1;
            for &(dx, dy) in &MOVES {
                let nx = cx + dx;
                let ny = cy + dy;
                if nx >= 0 && nx < GRID_SIZE && ny >= 0 && ny < GRID_SIZE {
                    let idx = (ny * GRID_SIZE + nx) as usize;
                    if !self.bfs_visited[idx] {
                        self.bfs_visited[idx] = true;
                        self.bfs_queue.push_back((nx, ny));
                    }
                }
            }
        }
        count as f32 / total_empty as f32
    }

    fn is_collision(&self, x: i32, y: i32) -> bool {
        if x < 0 || x >= GRID_SIZE || y < 0 || y >= GRID_SIZE { return true; }
        if self.snake_set[(y * GRID_SIZE + x) as usize] {
            if (x, y) == *self.snake.back().unwrap() { return false; }
            return true;
        }
        false
    }
    
    fn step(&mut self, action: i32) -> (f32, bool, i32) {
        self.steps_since_food += 1;
        let (head_x, head_y) = self.snake[0];
        if action == 1 { self.direction = (self.direction + 1) % 4; }
        else if action == 2 { self.direction = (self.direction + 3) % 4; }
        let (dx, dy) = MOVES[self.direction as usize];
        let new_head = (head_x + dx, head_y + dy);
        let (nx, ny) = new_head;
        if nx < 0 || nx >= GRID_SIZE || ny < 0 || ny >= GRID_SIZE || 
           (self.snake_set[(ny * GRID_SIZE + nx) as usize] && new_head != *self.snake.back().unwrap()) {
            return (-10.0, true, self.score);
        }
        self.snake.push_front(new_head);
        self.snake_set[(ny * GRID_SIZE + nx) as usize] = true;
        let mut reward = 0.0;
        let done = false; 
        let mut ate_food = false;
        if let Some(food_pos) = self.food {
            if new_head == food_pos {
                self.score += 1; reward = 10.0; self.steps_since_food = 0; self.place_food(); ate_food = true;
            }
        }
        if !ate_food {
            let tail = self.snake.pop_back().unwrap();
            self.snake_set[(tail.1 * GRID_SIZE + tail.0) as usize] = false;
            if let Some((fx, fy)) = self.food {
                let old_dist = (head_x - fx).abs() + (head_y - fy).abs();
                let new_dist = (nx - fx).abs() + (ny - fy).abs();
                if new_dist < old_dist { reward = 0.1; } else { reward = -0.15; }
            }
        }
        let max_steps = (GRID_SIZE * GRID_SIZE).max(self.snake.len() as i32 * MAX_STEPS_RATIO);
        if self.steps_since_food > max_steps { return (-10.0, true, self.score); }
        (reward, done, self.score)
    }

    fn get_state(&mut self) -> (Vec<f32>, Vec<f32>) {
        let mut spatial = vec![0.0f32; (3 * GRID_SIZE * GRID_SIZE) as usize];
        let (hx, hy) = self.snake[0];
        spatial[(0 * 400 + hy * 20 + hx) as usize] = 1.0;
        let snake_len = self.snake.len();
        for (i, &(x, y)) in self.snake.iter().skip(1).enumerate() {
            let val = 1.0 - (i as f32 / (snake_len as f32 + 1.0)) * 0.5;
            spatial[(1 * 400 + y * 20 + x) as usize] = val;
        }
        if let Some((fx, fy)) = self.food { spatial[(2 * 400 + fy * 20 + fx) as usize] = 1.0; }
        let mut aux = vec![0.0f32; 21];
        aux[self.direction as usize] = 1.0;
        for depth in 1..=3 {
            let base = 4 + (depth - 1) * 3;
            let d_i32 = depth as i32;
            let (dx, dy) = MOVES[self.direction as usize];
            aux[base] = if self.is_collision(hx + dx * d_i32, hy + dy * d_i32) { 1.0 } else { 0.0 };
            let (dx, dy) = MOVES[((self.direction + 1) % 4) as usize];
            aux[base + 1] = if self.is_collision(hx + dx * d_i32, hy + dy * d_i32) { 1.0 } else { 0.0 };
            let (dx, dy) = MOVES[((self.direction + 3) % 4) as usize];
            aux[base + 2] = if self.is_collision(hx + dx * d_i32, hy + dy * d_i32) { 1.0 } else { 0.0 };
        }
        if let Some((fx, fy)) = self.food {
            aux[13] = if fy < hy { 1.0 } else { 0.0 };
            aux[14] = if fy > hy { 1.0 } else { 0.0 };
            aux[15] = if fx < hx { 1.0 } else { 0.0 };
            aux[16] = if fx > hx { 1.0 } else { 0.0 };
        }
        aux[17] = snake_len as f32 / 400.0;
        let dirs = [self.direction, (self.direction + 1) % 4, (self.direction + 3) % 4];
        for (i, &d) in dirs.iter().enumerate() {
            let (dx, dy) = MOVES[d as usize];
            let nx = hx + dx;
            let ny = hy + dy;
            aux[18 + i] = self.flood_fill_count(nx, ny);
        }
        (spatial, aux)
    }
}

#[pyclass]
struct VectorizedSnakeEnv {
    games: Vec<SnakeGame>,
    n_envs: usize,
}

#[pymethods]
impl VectorizedSnakeEnv {
    #[new]
    fn new(n_envs: usize) -> Self {
        let mut games = Vec::with_capacity(n_envs);
        for _ in 0..n_envs { games.push(SnakeGame::new()); }
        VectorizedSnakeEnv { games, n_envs }
    }

    fn reset_all(&mut self) {
        self.games.par_iter_mut().for_each(|game| game.reset());
    }

    fn step_batch(&mut self, actions: Vec<i32>) -> (Vec<f32>, Vec<bool>, Vec<f32>) {
        let results: Vec<(f32, bool, i32)> = self.games.par_iter_mut()
            .zip(actions.par_iter())
            .map(|(game, &action)| {
                let res = game.step(action);
                if res.1 { game.reset(); }
                res
            })
            .collect();

        let mut rewards = Vec::with_capacity(self.n_envs);
        let mut dones = Vec::with_capacity(self.n_envs);
        let mut scores = Vec::with_capacity(self.n_envs);
        for (r, d, s) in results {
            rewards.push(r); dones.push(d); scores.push(s as f32);
        }
        (rewards, dones, scores)
    }

    fn get_states_batch<'py>(&mut self, py: Python<'py>) -> (&'py PyArray4<f32>, &'py PyArray2<f32>) {
        let states: Vec<(Vec<f32>, Vec<f32>)> = self.games.par_iter_mut()
            .map(|game| game.get_state())
            .collect();

        let mut spatial_batch = Vec::with_capacity(self.n_envs * 3 * 400);
        let mut aux_batch = Vec::with_capacity(self.n_envs * 21);
        for (s, a) in states {
            spatial_batch.extend(s);
            aux_batch.extend(a);
        }

        let spatial_flat = PyArray1::from_vec(py, spatial_batch);
        let spatial_reshaped = spatial_flat.reshape((self.n_envs, 3, 20, 20)).unwrap();
        
        let spatial_final: &PyArray4<f32> = spatial_reshaped.downcast().unwrap();

        let aux_flat = PyArray1::from_vec(py, aux_batch);
        let aux_reshaped = aux_flat.reshape((self.n_envs, 21)).unwrap();
        let aux_final: &PyArray2<f32> = aux_reshaped.downcast().unwrap();

        (spatial_final, aux_final)
    }
}

#[pymodule]
fn fast_snake_v6(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<VectorizedSnakeEnv>()?;
    Ok(())
}
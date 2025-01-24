import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

const generateData = () => {
  const data = [];
  let cumSum = 0;
  for (let i = 1; i <= 1000; i++) {
    cumSum += Math.random() - 0.5;
    data.push({
      trials: i,
      average: cumSum / i
    });
  }
  return data;
};

const ConvergenceChart = () => {
  const data = generateData();

  return (
    <ResponsiveContainer width="100%" height={300}>
      <LineChart data={data}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="trials" type="number" scale="log" domain={['auto', 'auto']} />
        <YAxis />
        <Tooltip />
        <Legend />
        <Line type="monotone" dataKey="average" stroke="#8884d8" dot={false} />
      </LineChart>
    </ResponsiveContainer>
  );
};

export default ConvergenceChart;
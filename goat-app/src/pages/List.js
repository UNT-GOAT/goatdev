import React from 'react';

function List({ goatData, handleSort, sortOption }) {
  return (
    <div className="list-page">
      <h1>Goat List</h1>
      <div className="search-controls">
        <label>
          Sort by:
          <select
            value={sortOption}
            onChange={(e) => handleSort(e.target.value)}
          >
            <option value="">Select</option>
            <option value="weight-asc">Weight (Ascending)</option>
            <option value="weight-desc">Weight (Descending)</option>
            <option value="age-asc">Age (Ascending)</option>
            <option value="age-desc">Age (Descending)</option>
            <option value="grade-asc">Grade (Ascending)</option>
            <option value="grade-desc">Grade (Descending)</option>
          </select>
        </label>
      </div>
      <table className="goat-table">
        <thead>
          <tr>
            <th>ID</th>
            <th>Name</th>
            <th>Weight</th>
            <th>Age</th>
            <th>Grade</th>
          </tr>
        </thead>
        <tbody>
          {goatData.map((goat) => (
            <tr key={goat.id}>
              <td>{goat.id}</td>
              <td>{goat.name}</td>
              <td>{goat.weight} kg</td>
              <td>{goat.age} years</td>
              <td>{goat.grade}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default List;
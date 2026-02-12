import React from 'react';

function Search({ searchQuery, handleSearchInput, searchAttribute, setSearchAttribute, filteredGoats }) {
  return (
    <div className="search-page">
      <h1>Search Goats</h1>
      <div className="search-controls">
        <label>
          Search by:
          <select
            value={searchAttribute}
            onChange={(e) => {
              setSearchAttribute(e.target.value);
              handleSearchInput({ target: { value: '' } }); // Clear search query when changing attribute
            }}
          >
            <option value="id">ID</option>
            <option value="name">Name</option>
            <option value="weight">Weight</option>
            <option value="age">Age</option>
            <option value="grade">Grade</option>
          </select>
        </label>
        <input
          type="text"
          placeholder={`Enter ${searchAttribute}`}
          value={searchQuery}
          onChange={handleSearchInput}
        />
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
          {filteredGoats.map((goat) => (
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

export default Search;
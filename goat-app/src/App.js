import React, { useState } from 'react';
import Login from './pages/Login';
import List from './pages/List';
import Search from './pages/Search';
import './App.css';

function App() {
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [isLogin, setIsLogin] = useState(true);
  const [currentPage, setCurrentPage] = useState('list');
  const [goatData, setGoatData] = useState([
    { id: 1, name: 'Goat A', weight: 50, age: 2, grade: 'A' },
    { id: 2, name: 'Goat B', weight: 45, age: 1.5, grade: 'B' },
    { id: 3, name: 'Goat C', weight: 55, age: 3, grade: 'A' },
  ]);
  const [searchQuery, setSearchQuery] = useState('');
  const [searchAttribute, setSearchAttribute] = useState('id');
  const [sortOption, setSortOption] = useState('');

  const handleLogin = (username, password) => {
    if (username === 'johngoat' && password === 'herdsync') {
      setIsLoggedIn(true);
    } else {
      alert('Invalid username or password');
    }
  };

  const handleSort = (option) => {
    const sortedData = [...goatData];
    if (option === 'weight-asc') {
      sortedData.sort((a, b) => a.weight - b.weight);
    } else if (option === 'weight-desc') {
      sortedData.sort((a, b) => b.weight - a.weight);
    } else if (option === 'age-asc') {
      sortedData.sort((a, b) => a.age - b.age);
    } else if (option === 'age-desc') {
      sortedData.sort((a, b) => b.age - a.age);
    } else if (option === 'grade-asc') {
      sortedData.sort((a, b) => a.grade.localeCompare(b.grade));
    } else if (option === 'grade-desc') {
      sortedData.sort((a, b) => b.grade.localeCompare(a.grade));
    }
    setGoatData(sortedData);
    setSortOption(option);
  };

  const handleSearchInput = (e) => {
    const value = e.target.value;
    if (searchAttribute === 'id' || searchAttribute === 'weight' || searchAttribute === 'age') {
      if (/^\d*$/.test(value)) {
        setSearchQuery(value);
      }
    } else {
      setSearchQuery(value);
    }
  };

  if (!isLoggedIn) {
    return (
      <Login
        onLogin={handleLogin}
        isLogin={isLogin}
        toggleLogin={() => setIsLogin(!isLogin)}
      />
    );
  }

  return (
    <div className="App">
      <nav className="navbar">
        <button onClick={() => setCurrentPage('list')}>List</button>
        <button onClick={() => setCurrentPage('search')}>Search</button>
      </nav>
      {currentPage === 'list' && (
        <List
          goatData={goatData}
          handleSort={handleSort}
          sortOption={sortOption}
        />
      )}
      {currentPage === 'search' && (
        <Search
          searchQuery={searchQuery}
          handleSearchInput={handleSearchInput}
          searchAttribute={searchAttribute}
          setSearchAttribute={setSearchAttribute}
          filteredGoats={goatData.filter((goat) => {
            if (searchAttribute === 'id') {
              return goat.id.toString().includes(searchQuery);
            } else if (searchAttribute === 'weight') {
              return goat.weight.toString().includes(searchQuery);
            } else if (searchAttribute === 'age') {
              return goat.age.toString().includes(searchQuery);
            } else if (searchAttribute === 'grade') {
              return goat.grade.toLowerCase().includes(searchQuery.toLowerCase());
            } else if (searchAttribute === 'name') {
              return goat.name.toLowerCase().includes(searchQuery.toLowerCase());
            }
            return false;
          })}
        />
      )}
    </div>
  );
}

export default App;
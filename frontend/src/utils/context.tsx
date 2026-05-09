import React, { createContext, useContext, useReducer, ReactNode, useMemo } from 'react';
import { ComparisonData, WatchlistItem, UserPreferences } from '../types';
import { AnalysisResult } from '../types/backend';

interface AppState {
  currentStock: AnalysisResult | null;
  comparisonData: ComparisonData | null;
  watchlist: WatchlistItem[];
  userPreferences: UserPreferences;
  isComparisonMode: boolean;
  recentSearches: string[];
}

type AppAction =
  | { type: 'SET_CURRENT_STOCK'; payload: AnalysisResult }
  | { type: 'SET_COMPARISON_DATA'; payload: ComparisonData }
  | { type: 'SET_WATCHLIST'; payload: WatchlistItem[] }
  | { type: 'ADD_TO_WATCHLIST'; payload: WatchlistItem }
  | { type: 'REMOVE_FROM_WATCHLIST'; payload: string }
  | { type: 'SET_USER_PREFERENCES'; payload: UserPreferences }
  | { type: 'TOGGLE_COMPARISON_MODE' }
  | { type: 'ADD_RECENT_SEARCH'; payload: string }
  | { type: 'CLEAR_CURRENT_STOCK' }
  | { type: 'CLEAR_COMPARISON_DATA' };

const initialState: AppState = {
  currentStock: null,
  comparisonData: null,
  watchlist: [],
  userPreferences: {
    default_time_range: '1M',
    default_indicators: ['RSI', 'MACD', 'SMA'],
    alert_preferences: {
      email_notifications: true,
      push_notifications: false,
      price_movement_threshold: 5,
    },
    theme: 'light',
  },
  isComparisonMode: false,
  recentSearches: [],
};

function appReducer(state: AppState, action: AppAction): AppState {
  switch (action.type) {
    case 'SET_CURRENT_STOCK':
      return {
        ...state,
        currentStock: action.payload,
      };

    case 'SET_COMPARISON_DATA':
      return {
        ...state,
        comparisonData: action.payload,
      };

    case 'SET_WATCHLIST':
      return {
        ...state,
        watchlist: action.payload,
      };

    case 'ADD_TO_WATCHLIST':
      return {
        ...state,
        watchlist: [...state.watchlist, action.payload],
      };

    case 'REMOVE_FROM_WATCHLIST':
      return {
        ...state,
        watchlist: state.watchlist.filter(item => item.ticker !== action.payload),
      };

    case 'SET_USER_PREFERENCES':
      return {
        ...state,
        userPreferences: action.payload,
      };

    case 'TOGGLE_COMPARISON_MODE':
      return {
        ...state,
        isComparisonMode: !state.isComparisonMode,
        comparisonData: !state.isComparisonMode ? state.comparisonData : null,
      };

    case 'ADD_RECENT_SEARCH':
      const newSearches = [action.payload, ...state.recentSearches.filter(s => s !== action.payload)].slice(0, 5);
      return {
        ...state,
        recentSearches: newSearches,
      };

    case 'CLEAR_CURRENT_STOCK':
      return {
        ...state,
        currentStock: null,
      };

    case 'CLEAR_COMPARISON_DATA':
      return {
        ...state,
        comparisonData: null,
      };

    default:
      return state;
  }
}

interface AppContextType {
  state: AppState;
  dispatch: React.Dispatch<AppAction>;
  actions: {
    setCurrentStock: (stock: AnalysisResult) => void;
    setComparisonData: (data: ComparisonData) => void;
    setWatchlist: (watchlist: WatchlistItem[]) => void;
    addToWatchlist: (item: WatchlistItem) => void;
    removeFromWatchlist: (ticker: string) => void;
    setUserPreferences: (preferences: UserPreferences) => void;
    toggleComparisonMode: () => void;
    addRecentSearch: (ticker: string) => void;
    clearCurrentStock: () => void;
    clearComparisonData: () => void;
  };
}

const AppContext = createContext<AppContextType | undefined>(undefined);

interface AppProviderProps {
  children: ReactNode;
}

export const AppProvider: React.FC<AppProviderProps> = ({ children }) => {
  const [state, dispatch] = useReducer(appReducer, initialState);

  const actions = useMemo(() => ({
    setCurrentStock: (stock: AnalysisResult) => {
      dispatch({ type: 'SET_CURRENT_STOCK', payload: stock });
      dispatch({ type: 'ADD_RECENT_SEARCH', payload: stock.stock_data.ticker });
    },
    setComparisonData: (data: ComparisonData) => {
      dispatch({ type: 'SET_COMPARISON_DATA', payload: data });
    },
    setWatchlist: (watchlist: WatchlistItem[]) => {
      dispatch({ type: 'SET_WATCHLIST', payload: watchlist });
    },
    addToWatchlist: (item: WatchlistItem) => {
      dispatch({ type: 'ADD_TO_WATCHLIST', payload: item });
    },
    removeFromWatchlist: (ticker: string) => {
      dispatch({ type: 'REMOVE_FROM_WATCHLIST', payload: ticker });
    },
    setUserPreferences: (preferences: UserPreferences) => {
      dispatch({ type: 'SET_USER_PREFERENCES', payload: preferences });
    },
    toggleComparisonMode: () => {
      dispatch({ type: 'TOGGLE_COMPARISON_MODE' });
    },
    addRecentSearch: (ticker: string) => {
      dispatch({ type: 'ADD_RECENT_SEARCH', payload: ticker });
    },
    clearCurrentStock: () => {
      dispatch({ type: 'CLEAR_CURRENT_STOCK' });
    },
    clearComparisonData: () => {
      dispatch({ type: 'CLEAR_COMPARISON_DATA' });
    },
  }), []);

  const contextValue = useMemo(() => ({
    state,
    dispatch,
    actions
  }), [state, actions]);

  return (
    <AppContext.Provider value={contextValue}>
      {children}
    </AppContext.Provider>
  );
};

export const useAppContext = () => {
  const context = useContext(AppContext);
  if (context === undefined) {
    throw new Error('useAppContext must be used within an AppProvider');
  }
  return context;
};

export const useAppState = () => {
  const { state } = useAppContext();
  return state;
};

export const useAppActions = () => {
  const { actions } = useAppContext();
  return actions;
};
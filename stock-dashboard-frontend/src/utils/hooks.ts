import { useState, useEffect, useCallback } from 'react';
import { APIError, LoadingState } from '../types';
import { AnalysisResult } from '../types/backend';
import apiService from '../services/api';

export function useLoadingState(initialState: boolean = false): [LoadingState, (isLoading: boolean, error?: APIError | null) => void] {
  const [state, setState] = useState<LoadingState>({
    isLoading: initialState,
    error: null,
  });

  const setLoadingState = useCallback((isLoading: boolean, error: APIError | null = null) => {
    setState({ isLoading, error });
  }, []);

  return [state, setLoadingState];
}

export function useAsyncOperation<T>() {
  const [loadingState, setLoadingState] = useLoadingState();
  const [data, setData] = useState<T | null>(null);

  const execute = useCallback(async (operation: () => Promise<T>) => {
    try {
      setLoadingState(true);
      const result = await operation();
      setData(result);
      setLoadingState(false);
      return result;
    } catch (error) {
      const apiError = error as APIError;
      setLoadingState(false, apiError);
      throw error;
    }
  }, [setLoadingState]);

  const reset = useCallback(() => {
    setData(null);
    setLoadingState(false);
  }, [setLoadingState]);

  return {
    ...loadingState,
    data,
    execute,
    reset,
  };
}

export function useStockAnalysis() {
  const operation = useAsyncOperation<AnalysisResult>();

  const analyzeStock = useCallback(async (ticker: string, onProgress?: (progress: number, step: string) => void) => {
    return operation.execute(() => apiService.analyzeStock(ticker, onProgress));
  }, [operation]);

  return {
    ...operation,
    analyzeStock,
  };
}

export function useStockData() {
  const operation = useAsyncOperation();

  const fetchStockData = useCallback(async (ticker: string) => {
    return operation.execute(() => apiService.getStockData(ticker));
  }, [operation]);

  return {
    ...operation,
    fetchStockData,
  };
}

export function useStockComparison() {
  const operation = useAsyncOperation();

  const compareStocks = useCallback(async (primaryTicker: string, comparisonTicker: string) => {
    return operation.execute(() => apiService.compareStocks(primaryTicker, comparisonTicker));
  }, [operation]);

  return {
    ...operation,
    compareStocks,
  };
}

export function useStockSearch() {
  const [searchResults, setSearchResults] = useState<any[]>([]);
  const [loadingState, setLoadingState] = useLoadingState();

  const search = useCallback(async (query: string) => {
    if (!query.trim()) {
      setSearchResults([]);
      return;
    }

    try {
      setLoadingState(true);
      const results = await apiService.searchStocks(query);
      setSearchResults(results);
      setLoadingState(false);
    } catch (error) {
      const apiError = error as APIError;
      setLoadingState(false, apiError);
      setSearchResults([]);
    }
  }, [setLoadingState]);

  const clearResults = useCallback(() => {
    setSearchResults([]);
    setLoadingState(false);
  }, [setLoadingState]);

  return {
    searchResults,
    search,
    clearResults,
    ...loadingState,
  };
}

export function useWatchlist() {
  const [watchlist, setWatchlist] = useState<any[]>([]);
  const [loadingState, setLoadingState] = useLoadingState();

  const fetchWatchlist = useCallback(async () => {
    try {
      setLoadingState(true);
      const data = await apiService.getWatchlist();
      setWatchlist(data);
      setLoadingState(false);
    } catch (error) {
      const apiError = error as APIError;
      setLoadingState(false, apiError);
    }
  }, [setLoadingState]);

  const addToWatchlist = useCallback(async (ticker: string) => {
    try {
      await apiService.addToWatchlist(ticker);
      await fetchWatchlist(); // Refresh the list
    } catch (error) {
      throw error;
    }
  }, [fetchWatchlist]);

  const removeFromWatchlist = useCallback(async (ticker: string) => {
    try {
      await apiService.removeFromWatchlist(ticker);
      await fetchWatchlist(); // Refresh the list
    } catch (error) {
      throw error;
    }
  }, [fetchWatchlist]);

  useEffect(() => {
    fetchWatchlist();
  }, [fetchWatchlist]);

  return {
    watchlist,
    addToWatchlist,
    removeFromWatchlist,
    refreshWatchlist: fetchWatchlist,
    ...loadingState,
  };
}

export function useDebounce<T>(value: T, delay: number): T {
  const [debouncedValue, setDebouncedValue] = useState<T>(value);

  useEffect(() => {
    const handler = setTimeout(() => {
      setDebouncedValue(value);
    }, delay);

    return () => {
      clearTimeout(handler);
    };
  }, [value, delay]);

  return debouncedValue;
}
import io
import sys
from unittest.mock import patch, MagicMock, mock_open
import arxiv
from main import (
    save_faiss_index,
    load_faiss_index,
    save_docstore_and_mapping,
    load_docstore_and_mapping,
    fetch_papers,
    summarize_paper,
    store_summary,
    search_summaries,
    main
)

# Test saving FAISS index
@patch('faiss.write_index')
def test_save_faiss_index(mock_write_index):
    index = MagicMock()
    file_path = "test_index.bin"
    save_faiss_index(index, file_path)
    mock_write_index.assert_called_once_with(index, file_path)

# Test loading FAISS index when file exists
@patch('os.path.exists', return_value=True)
@patch('faiss.read_index')
def test_load_faiss_index_existing(mock_read_index, mock_exists):
    file_path = "test_index.bin"
    load_faiss_index(file_path)
    mock_exists.assert_called_once_with(file_path)
    mock_read_index.assert_called_once_with(file_path)

# Test loading FAISS index when file doesn't exist
def test_load_faiss_index_nonexisting():
    with patch('os.path.exists', return_value=False) as mock_exists, \
         patch('faiss.IndexFlatL2') as mock_index, \
         patch('main.OpenAIEmbeddings') as mock_embeddings_class:
        
        # Set up the mock for OpenAIEmbeddings()
        mock_embeddings = MagicMock()
        mock_embeddings.embed_query.return_value = [0.1] * 5
        mock_embeddings_class.return_value = mock_embeddings
        
        file_path = "nonexistent_index.bin"
        
        # Reset the mock before the test to clear any previous calls
        mock_exists.reset_mock()
        
        load_faiss_index(file_path)
        
        # Check exists was called with our file path
        mock_exists.assert_any_call(file_path)
        
        # Check the embedding and index were created correctly
        mock_embeddings.embed_query.assert_called_with("hello world")
        mock_index.assert_called_once()

# Test saving docstore and mapping
@patch('builtins.open', new_callable=mock_open)
@patch('pickle.dump')
def test_save_docstore_and_mapping(mock_dump, mock_file):
    docstore = {"doc1": "content1"}
    index_to_docstore_id = {0: "doc1"}
    docstore_path = "test_docstore.pkl"
    mapping_path = "test_mapping.pkl"
    
    save_docstore_and_mapping(docstore, index_to_docstore_id, docstore_path, mapping_path)
    
    assert mock_file.call_count == 2
    assert mock_dump.call_count == 2
    mock_dump.assert_any_call(docstore, mock_file())
    mock_dump.assert_any_call(index_to_docstore_id, mock_file())

# Test loading docstore and mapping when files exist
@patch('os.path.exists', return_value=True)
@patch('builtins.open', new_callable=mock_open)
@patch('pickle.load')
def test_load_docstore_and_mapping_existing(mock_load, mock_file, mock_exists):
    mock_load.side_effect = [{"doc1": "content1"}, {0: "doc1"}]
    docstore_path = "test_docstore.pkl"
    mapping_path = "test_mapping.pkl"
    
    docstore, index_to_docstore_id = load_docstore_and_mapping(docstore_path, mapping_path)
    
    assert mock_exists.call_count == 2
    assert mock_file.call_count == 2
    assert mock_load.call_count == 2
    assert docstore == {"doc1": "content1"}
    assert index_to_docstore_id == {0: "doc1"}

# Test loading docstore and mapping when files don't exist
def test_load_docstore_and_mapping_nonexisting():
    # Looking at the implementation, the function checks:
    # if os.path.exists(docstore_path) and os.path.exists(mapping_path):
    
    with patch('os.path.exists') as mock_exists:
        # Make the check return False to enter the else branch
        mock_exists.return_value = False
        
        docstore_path = "nonexistent_docstore.pkl"
        mapping_path = "nonexistent_mapping.pkl"
        
        # In the implementation, it creates an InMemoryDocstore directly
        docstore, index_to_docstore_id = load_docstore_and_mapping(docstore_path, mapping_path)
        
        # Verify os.path.exists was called
        mock_exists.assert_called()
        
        # Verify the second return value is an empty dict
        assert isinstance(index_to_docstore_id, dict)
        assert len(index_to_docstore_id) == 0
        
        # For the docstore object, verify it's an instance of InMemoryDocstore
        from langchain_community.docstore.in_memory import InMemoryDocstore
        assert isinstance(docstore, InMemoryDocstore)

# Test fetching papers from ArXiv
@patch('arxiv.Search')
def test_fetch_papers(mock_search):
    mock_results = [MagicMock(), MagicMock()]
    mock_search.return_value.results.return_value = mock_results
    
    query = "machine learning"
    max_results = 2
    results = fetch_papers(query, max_results)
    
    mock_search.assert_called_once_with(
        query=query, 
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )
    assert results == mock_results

# Test summarizing a paper
@patch('langchain_community.adapters.openai.ChatCompletion.create')
def test_summarize_paper(mock_create):
    paper = MagicMock()
    paper.title = "Test Paper"
    paper.summary = "This is a test abstract"
    
    mock_create.return_value = {
        "choices": [{"message": {"content": "Test summary"}}]
    }
    
    result = summarize_paper(paper)
    
    mock_create.assert_called_once()
    assert result == "Test summary"

# Test storing a summary
@patch('main.save_faiss_index')
@patch('main.save_docstore_and_mapping')
def test_store_summary(mock_save_docstore, mock_save_index, monkeypatch):
    # Mock vector_store
    mock_vector_store = MagicMock()
    monkeypatch.setattr('main.vector_store', mock_vector_store)
    monkeypatch.setattr('main.index', MagicMock())
    monkeypatch.setattr('main.docstore', MagicMock())
    monkeypatch.setattr('main.index_to_docstore_id', MagicMock())
    
    title = "Test Paper"
    summary = "This is a test summary"
    
    store_summary(title, summary)
    
    mock_vector_store.add_documents.assert_called_once()
    mock_save_index.assert_called_once()
    mock_save_docstore.assert_called_once()

# Test searching summaries
def test_search_summaries(monkeypatch):
    # Mock expected results
    mock_results = [MagicMock(), MagicMock(), MagicMock()]
    
    # Mock vector_store
    mock_vector_store = MagicMock()
    mock_vector_store.similarity_search.return_value = mock_results
    monkeypatch.setattr('main.vector_store', mock_vector_store)
    
    query = "machine learning"
    top_k = 3
    
    results = search_summaries(query, top_k)
    
    mock_vector_store.similarity_search.assert_called_once_with(query, k=top_k)
    assert results == mock_results

# Test main function - research option
@patch('builtins.input', side_effect=['1', 'machine learning', '3'])
@patch('main.fetch_papers')
@patch('main.summarize_paper')
@patch('main.store_summary')
def test_main_research_option(mock_store, mock_summarize, mock_fetch, mock_input):
    # Mock papers
    paper1 = MagicMock()
    paper1.title = "Paper 1"
    paper2 = MagicMock()
    paper2.title = "Paper 2"
    mock_fetch.return_value = [paper1, paper2]
    
    # Mock summaries
    mock_summarize.side_effect = ["Summary 1", "Summary 2"]
    
    # Redirect stdout to capture print statements
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    try:
        main()
    finally:
        sys.stdout = sys.__stdout__
    
    # Verify function calls
    mock_fetch.assert_called_once_with('machine learning')
    assert mock_summarize.call_count == 2
    assert mock_store.call_count == 2
    mock_store.assert_any_call("Paper 1", "Summary 1")
    mock_store.assert_any_call("Paper 2", "Summary 2")
    
    # Verify expected output was printed
    output = captured_output.getvalue()
    assert "Summarizing: Paper 1" in output
    assert "Summarizing: Paper 2" in output
    assert "Stored summaries are ready for retrieval" in output

# Test main function - search option
@patch('builtins.input', side_effect=['2', 'neural networks', '3'])
@patch('main.search_summaries')
def test_main_search_option(mock_search, mock_input):
    # Mock search results
    doc1 = MagicMock()
    doc1.metadata = {'title': 'Neural Networks Paper'}
    doc1.page_content = 'Content about neural networks'
    
    doc2 = MagicMock()
    doc2.metadata = {'title': 'Deep Learning Paper'}
    doc2.page_content = 'Content about deep learning'
    
    mock_search.return_value = [doc1, doc2]
    
    # Redirect stdout to capture print statements
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    try:
        main()
    finally:
        sys.stdout = sys.__stdout__
    
    # Verify function calls
    mock_search.assert_called_once_with('neural networks')
    
    # Verify expected output was printed
    output = captured_output.getvalue()
    assert "Search Results:" in output
    assert "1. Neural Networks Paper" in output
    assert "Content about neural networks" in output
    assert "2. Deep Learning Paper" in output
    assert "Content about deep learning" in output

# Test main function - invalid option
@patch('builtins.input', side_effect=['4', '3'])
def test_main_invalid_option(mock_input):
    # Redirect stdout to capture print statements
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    try:
        main()
    finally:
        sys.stdout = sys.__stdout__
    
    # Verify expected output was printed
    output = captured_output.getvalue()
    assert "Invalid choice. Please select 1, 2, or 3." in output
"""
Content Intelligence Service
Implements vector database content intelligence with ChromaDB integration for pattern analysis and predictive quality scoring.
"""

import os
import json
import uuid
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from loguru import logger
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from app.config import settings


@dataclass
class ContentPattern:
    """Represents a content pattern identified by the intelligence system."""
    pattern_id: str
    pattern_type: str
    description: str
    frequency: int
    quality_score: float
    success_rate: float
    examples: List[str]
    metadata: Dict[str, Any]


@dataclass
class QualityPrediction:
    """Represents a quality prediction for content."""
    content_id: str
    predicted_quality: float
    confidence: float
    factors: Dict[str, float]
    recommendations: List[str]
    risk_factors: List[str]


@dataclass
class PerformanceMetrics:
    """Represents performance metrics for content processing."""
    timestamp: datetime
    job_id: str
    processing_time: float
    quality_score: float
    error_count: int
    human_reviews: int
    success: bool
    content_type: str
    patterns_used: List[str]


class ContentIntelligence:
    """
    Content Intelligence Service with ChromaDB integration.
    
    Features:
    - Semantic content search and retrieval
    - Pattern recognition and analysis
    - Performance tracking and metrics
    - Predictive quality scoring
    - Adaptive processing strategies
    - Cross-document learning insights
    """
    
    def __init__(self):
        """Initialize the content intelligence service."""
        self.intelligence_db_path = os.path.join(settings.temp_dir, "content_intelligence")
        self.embeddings_model = "all-MiniLM-L6-v2"
        self.pattern_collection_name = "content_patterns"
        self.performance_collection_name = "performance_metrics"
        self.quality_collection_name = "quality_predictions"
        
        # Initialize components
        self._initialize_chromadb()
        self._initialize_embeddings()
        self._initialize_analyzers()
        
        logger.info("Content Intelligence Service initialized")
    
    def _initialize_chromadb(self):
        """Initialize ChromaDB for content intelligence."""
        try:
            os.makedirs(self.intelligence_db_path, exist_ok=True)
            
            # Initialize ChromaDB client
            self.chroma_client = chromadb.PersistentClient(
                path=self.intelligence_db_path,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Create collections for different data types
            self.pattern_collection = self.chroma_client.get_or_create_collection(
                name=self.pattern_collection_name,
                metadata={"description": "Content patterns and insights"}
            )
            
            self.performance_collection = self.chroma_client.get_or_create_collection(
                name=self.performance_collection_name,
                metadata={"description": "Performance metrics and analytics"}
            )
            
            self.quality_collection = self.chroma_client.get_or_create_collection(
                name=self.quality_collection_name,
                metadata={"description": "Quality predictions and assessments"}
            )
            
            logger.info("ChromaDB collections initialized for content intelligence")
            
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {str(e)}")
            raise
    
    def _initialize_embeddings(self):
        """Initialize embedding model for content analysis."""
        try:
            self.embeddings = SentenceTransformer(self.embeddings_model)
            logger.info(f"Embeddings model '{self.embeddings_model}' initialized")
        except Exception as e:
            logger.error(f"Failed to initialize embeddings: {str(e)}")
            raise
    
    def _initialize_analyzers(self):
        """Initialize text analysis components."""
        try:
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2)
            )
            self.kmeans_clusterer = KMeans(n_clusters=5, random_state=42)
            logger.info("Text analyzers initialized")
        except Exception as e:
            logger.error(f"Failed to initialize analyzers: {str(e)}")
            raise
    
    async def analyze_content_patterns(
        self, 
        content: str, 
        job_id: str,
        content_type: str = "educational"
    ) -> Dict[str, Any]:
        """
        Analyze content patterns and extract insights.
        
        Args:
            content: The content to analyze
            job_id: Unique job identifier
            content_type: Type of content being analyzed
            
        Returns:
            Pattern analysis results
        """
        try:
            logger.info(f"Analyzing content patterns for job {job_id}")
            
            # Extract text features
            text_features = await self._extract_text_features(content)
            
            # Find similar content patterns
            similar_patterns = await self._find_similar_patterns(content, content_type)
            
            # Identify new patterns
            new_patterns = await self._identify_new_patterns(content, text_features, job_id)
            
            # Generate insights
            insights = await self._generate_pattern_insights(similar_patterns, new_patterns)
            
            # Store patterns for future learning
            await self._store_patterns(new_patterns, job_id)
            
            logger.info(f"Pattern analysis completed for job {job_id}")
            
            return {
                "success": True,
                "job_id": job_id,
                "similar_patterns": similar_patterns,
                "new_patterns": new_patterns,
                "insights": insights,
                "text_features": text_features,
                "analysis_timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing content patterns: {str(e)}")
            return {
                "success": False,
                "job_id": job_id,
                "error": str(e)
            }
    
    async def _extract_text_features(self, content: str) -> Dict[str, Any]:
        """Extract text features for pattern analysis."""
        try:
            # Basic text statistics
            word_count = len(content.split())
            char_count = len(content)
            sentence_count = len(content.split('.'))
            
            # TF-IDF features
            tfidf_matrix = self.tfidf_vectorizer.fit_transform([content])
            feature_names = self.tfidf_vectorizer.get_feature_names_out()
            tfidf_scores = tfidf_matrix.toarray()[0]
            
            # Get top features
            top_features = sorted(
                zip(feature_names, tfidf_scores),
                key=lambda x: x[1],
                reverse=True
            )[:20]
            
            # Extract semantic features
            embedding = self.embeddings.encode([content])[0]
            
            return {
                "word_count": word_count,
                "char_count": char_count,
                "sentence_count": sentence_count,
                "top_features": top_features,
                "embedding": embedding.tolist(),
                "complexity_score": self._calculate_complexity_score(content),
                "readability_score": self._calculate_readability_score(content)
            }
            
        except Exception as e:
            logger.error(f"Error extracting text features: {str(e)}")
            return {}
    
    def _calculate_complexity_score(self, content: str) -> float:
        """Calculate content complexity score."""
        try:
            words = content.split()
            if not words:
                return 0.0
            
            # Average word length
            avg_word_length = sum(len(word) for word in words) / len(words)
            
            # Sentence complexity (words per sentence)
            sentences = content.split('.')
            avg_sentence_length = len(words) / len(sentences) if sentences else 0
            
            # Vocabulary diversity (unique words / total words)
            unique_words = len(set(word.lower() for word in words))
            vocabulary_diversity = unique_words / len(words) if words else 0
            
            # Combine factors
            complexity = (
                (avg_word_length / 10) * 0.3 +
                (avg_sentence_length / 20) * 0.4 +
                vocabulary_diversity * 0.3
            )
            
            return min(complexity, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating complexity score: {str(e)}")
            return 0.5
    
    def _calculate_readability_score(self, content: str) -> float:
        """Calculate content readability score."""
        try:
            words = content.split()
            sentences = content.split('.')
            
            if not words or not sentences:
                return 0.5
            
            # Simple readability formula
            avg_words_per_sentence = len(words) / len(sentences)
            avg_syllables_per_word = self._estimate_syllables_per_word(words)
            
            # Flesch Reading Ease approximation
            readability = 206.835 - (1.015 * avg_words_per_sentence) - (84.6 * avg_syllables_per_word)
            
            # Normalize to 0-1 scale
            return max(0, min(1, readability / 100))
            
        except Exception as e:
            logger.error(f"Error calculating readability score: {str(e)}")
            return 0.5
    
    def _estimate_syllables_per_word(self, words: List[str]) -> float:
        """Estimate average syllables per word."""
        try:
            total_syllables = 0
            for word in words:
                # Simple syllable estimation
                vowels = 'aeiouy'
                word = word.lower()
                syllable_count = 0
                prev_was_vowel = False
                
                for char in word:
                    is_vowel = char in vowels
                    if is_vowel and not prev_was_vowel:
                        syllable_count += 1
                    prev_was_vowel = is_vowel
                
                # Handle silent 'e'
                if word.endswith('e') and syllable_count > 1:
                    syllable_count -= 1
                
                total_syllables += max(1, syllable_count)
            
            return total_syllables / len(words) if words else 0
            
        except Exception as e:
            logger.error(f"Error estimating syllables: {str(e)}")
            return 2.0
    
    async def _find_similar_patterns(
        self, 
        content: str, 
        content_type: str
    ) -> List[Dict[str, Any]]:
        """Find similar content patterns in the database."""
        try:
            # Search for similar patterns
            results = self.pattern_collection.query(
                query_texts=[content],
                n_results=5,
                where={"content_type": content_type}
            )
            
            similar_patterns = []
            if results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    similar_patterns.append({
                        "pattern_id": results['ids'][0][i],
                        "content": doc,
                        "similarity_score": 1 - results['distances'][0][i],
                        "metadata": results['metadatas'][0][i] if results['metadatas'] else {}
                    })
            
            return similar_patterns
            
        except Exception as e:
            logger.error(f"Error finding similar patterns: {str(e)}")
            return []
    
    async def _identify_new_patterns(
        self, 
        content: str, 
        text_features: Dict[str, Any],
        job_id: str
    ) -> List[ContentPattern]:
        """Identify new patterns in the content."""
        try:
            patterns = []
            
            # Pattern 1: Content structure patterns
            structure_pattern = self._identify_structure_pattern(content)
            if structure_pattern:
                patterns.append(structure_pattern)
            
            # Pattern 2: Language complexity patterns
            complexity_pattern = self._identify_complexity_pattern(text_features)
            if complexity_pattern:
                patterns.append(complexity_pattern)
            
            # Pattern 3: Topic coherence patterns
            coherence_pattern = self._identify_coherence_pattern(content)
            if coherence_pattern:
                patterns.append(coherence_pattern)
            
            # Pattern 4: Learning objective patterns
            learning_pattern = self._identify_learning_pattern(content)
            if learning_pattern:
                patterns.append(learning_pattern)
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error identifying new patterns: {str(e)}")
            return []
    
    def _identify_structure_pattern(self, content: str) -> Optional[ContentPattern]:
        """Identify content structure patterns."""
        try:
            # Check for common structural elements
            has_introduction = any(phrase in content.lower() for phrase in [
                'introduction', 'overview', 'welcome', 'let\'s start'
            ])
            has_conclusion = any(phrase in content.lower() for phrase in [
                'conclusion', 'summary', 'in summary', 'to conclude'
            ])
            has_numbered_sections = bool(content.count('1.') + content.count('2.') + content.count('3.'))
            has_bullet_points = bool(content.count('•') + content.count('-') + content.count('*'))
            
            structure_score = sum([has_introduction, has_conclusion, has_numbered_sections, has_bullet_points]) / 4
            
            if structure_score > 0.5:
                return ContentPattern(
                    pattern_id=f"structure_{uuid.uuid4().hex[:8]}",
                    pattern_type="structure",
                    description=f"Content structure pattern (score: {structure_score:.2f})",
                    frequency=1,
                    quality_score=structure_score,
                    success_rate=0.8,
                    examples=[content[:200] + "..."],
                    metadata={
                        "has_introduction": has_introduction,
                        "has_conclusion": has_conclusion,
                        "has_numbered_sections": has_numbered_sections,
                        "has_bullet_points": has_bullet_points
                    }
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error identifying structure pattern: {str(e)}")
            return None
    
    def _identify_complexity_pattern(self, text_features: Dict[str, Any]) -> Optional[ContentPattern]:
        """Identify language complexity patterns."""
        try:
            complexity_score = text_features.get("complexity_score", 0.5)
            readability_score = text_features.get("readability_score", 0.5)
            
            if complexity_score > 0.7 or readability_score < 0.3:
                complexity_level = "high" if complexity_score > 0.7 else "low"
                
                return ContentPattern(
                    pattern_id=f"complexity_{uuid.uuid4().hex[:8]}",
                    pattern_type="complexity",
                    description=f"Language complexity pattern ({complexity_level})",
                    frequency=1,
                    quality_score=0.6,  # Moderate quality for complex content
                    success_rate=0.7,
                    examples=[],
                    metadata={
                        "complexity_score": complexity_score,
                        "readability_score": readability_score,
                        "complexity_level": complexity_level
                    }
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error identifying complexity pattern: {str(e)}")
            return None
    
    def _identify_coherence_pattern(self, content: str) -> Optional[ContentPattern]:
        """Identify topic coherence patterns."""
        try:
            # Simple coherence check based on topic consistency
            sentences = content.split('.')
            if len(sentences) < 3:
                return None
            
            # Check for topic consistency (simplified)
            topic_words = []
            for sentence in sentences[:5]:  # Check first 5 sentences
                words = sentence.lower().split()
                # Extract potential topic words (nouns, longer words)
                topic_words.extend([w for w in words if len(w) > 4 and w.isalpha()])
            
            # Calculate topic consistency
            if topic_words:
                unique_topics = len(set(topic_words))
                total_topics = len(topic_words)
                coherence_score = unique_topics / total_topics if total_topics > 0 else 0
                
                if coherence_score > 0.3:  # Good coherence
                    return ContentPattern(
                        pattern_id=f"coherence_{uuid.uuid4().hex[:8]}",
                        pattern_type="coherence",
                        description=f"Topic coherence pattern (score: {coherence_score:.2f})",
                        frequency=1,
                        quality_score=coherence_score,
                        success_rate=0.8,
                        examples=[],
                        metadata={
                            "coherence_score": coherence_score,
                            "unique_topics": unique_topics,
                            "total_topics": total_topics
                        }
                    )
            
            return None
            
        except Exception as e:
            logger.error(f"Error identifying coherence pattern: {str(e)}")
            return None
    
    def _identify_learning_pattern(self, content: str) -> Optional[ContentPattern]:
        """Identify learning objective patterns."""
        try:
            # Check for learning-related keywords
            learning_keywords = [
                'learn', 'understand', 'know', 'skill', 'ability',
                'objective', 'goal', 'outcome', 'competency',
                'practice', 'exercise', 'example', 'demonstrate'
            ]
            
            content_lower = content.lower()
            learning_mentions = sum(1 for keyword in learning_keywords if keyword in content_lower)
            
            if learning_mentions > 3:  # Good learning focus
                return ContentPattern(
                    pattern_id=f"learning_{uuid.uuid4().hex[:8]}",
                    pattern_type="learning",
                    description=f"Learning objective pattern ({learning_mentions} mentions)",
                    frequency=1,
                    quality_score=min(learning_mentions / 10, 1.0),
                    success_rate=0.9,
                    examples=[],
                    metadata={
                        "learning_mentions": learning_mentions,
                        "learning_keywords_found": [kw for kw in learning_keywords if kw in content_lower]
                    }
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error identifying learning pattern: {str(e)}")
            return None
    
    async def _generate_pattern_insights(
        self, 
        similar_patterns: List[Dict[str, Any]],
        new_patterns: List[ContentPattern]
    ) -> Dict[str, Any]:
        """Generate insights from pattern analysis."""
        try:
            insights = {
                "pattern_summary": {
                    "similar_patterns_found": len(similar_patterns),
                    "new_patterns_identified": len(new_patterns),
                    "total_patterns": len(similar_patterns) + len(new_patterns)
                },
                "quality_indicators": [],
                "recommendations": [],
                "risk_factors": []
            }
            
            # Analyze similar patterns
            if similar_patterns:
                avg_similarity = np.mean([p["similarity_score"] for p in similar_patterns])
                insights["pattern_summary"]["average_similarity"] = avg_similarity
                
                if avg_similarity > 0.8:
                    insights["quality_indicators"].append("High similarity to successful content")
                elif avg_similarity > 0.6:
                    insights["quality_indicators"].append("Moderate similarity to existing content")
                else:
                    insights["quality_indicators"].append("Low similarity - potentially unique content")
            
            # Analyze new patterns
            if new_patterns:
                pattern_types = [p.pattern_type for p in new_patterns]
                insights["pattern_summary"]["pattern_types"] = list(set(pattern_types))
                
                high_quality_patterns = [p for p in new_patterns if p.quality_score > 0.7]
                if high_quality_patterns:
                    insights["quality_indicators"].append(f"Found {len(high_quality_patterns)} high-quality patterns")
                
                # Generate recommendations based on patterns
                for pattern in new_patterns:
                    if pattern.pattern_type == "structure" and pattern.quality_score > 0.7:
                        insights["recommendations"].append("Good content structure - maintain this approach")
                    elif pattern.pattern_type == "complexity" and pattern.quality_score < 0.5:
                        insights["recommendations"].append("Consider simplifying language for better readability")
                    elif pattern.pattern_type == "learning" and pattern.quality_score > 0.8:
                        insights["recommendations"].append("Excellent learning focus - continue this approach")
            
            # Identify risk factors
            if not similar_patterns and not new_patterns:
                insights["risk_factors"].append("No clear patterns identified - content may be too unique or unstructured")
            
            return insights
            
        except Exception as e:
            logger.error(f"Error generating pattern insights: {str(e)}")
            return {}
    
    async def _store_patterns(self, patterns: List[ContentPattern], job_id: str):
        """Store patterns in the database for future learning."""
        try:
            if not patterns:
                return
            
            # Prepare data for storage
            documents = []
            metadatas = []
            ids = []
            
            for pattern in patterns:
                documents.append(pattern.description)
                metadatas.append({
                    "pattern_id": pattern.pattern_id,
                    "pattern_type": pattern.pattern_type,
                    "frequency": pattern.frequency,
                    "quality_score": pattern.quality_score,
                    "success_rate": pattern.success_rate,
                    "job_id": job_id,
                    "timestamp": datetime.utcnow().isoformat(),
                    **pattern.metadata
                })
                ids.append(pattern.pattern_id)
            
            # Store in ChromaDB
            self.pattern_collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f"Stored {len(patterns)} patterns for job {job_id}")
            
        except Exception as e:
            logger.error(f"Error storing patterns: {str(e)}")
    
    async def predict_content_quality(
        self, 
        content: str, 
        job_id: str,
        content_type: str = "educational"
    ) -> QualityPrediction:
        """
        Predict content quality using machine learning models.
        
        Args:
            content: The content to analyze
            job_id: Unique job identifier
            content_type: Type of content
            
        Returns:
            Quality prediction with confidence and recommendations
        """
        try:
            logger.info(f"Predicting content quality for job {job_id}")
            
            # Extract features for prediction
            text_features = await self._extract_text_features(content)
            
            # Analyze patterns
            pattern_analysis = await self.analyze_content_patterns(content, job_id, content_type)
            
            # Calculate quality factors
            quality_factors = self._calculate_quality_factors(text_features, pattern_analysis)
            
            # Generate prediction
            predicted_quality = self._calculate_predicted_quality(quality_factors)
            confidence = self._calculate_prediction_confidence(quality_factors)
            
            # Generate recommendations
            recommendations = self._generate_quality_recommendations(quality_factors)
            risk_factors = self._identify_risk_factors(quality_factors)
            
            prediction = QualityPrediction(
                content_id=job_id,
                predicted_quality=predicted_quality,
                confidence=confidence,
                factors=quality_factors,
                recommendations=recommendations,
                risk_factors=risk_factors
            )
            
            # Store prediction
            await self._store_quality_prediction(prediction)
            
            logger.info(f"Quality prediction completed for job {job_id}: {predicted_quality:.2f}")
            return prediction
            
        except Exception as e:
            logger.error(f"Error predicting content quality: {str(e)}")
            return QualityPrediction(
                content_id=job_id,
                predicted_quality=0.5,
                confidence=0.0,
                factors={},
                recommendations=["Quality prediction failed"],
                risk_factors=["Prediction error"]
            )
    
    def _calculate_quality_factors(
        self, 
        text_features: Dict[str, Any],
        pattern_analysis: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate quality factors for prediction."""
        try:
            factors = {}
            
            # Text quality factors
            factors["word_count_score"] = min(text_features.get("word_count", 0) / 1000, 1.0)
            factors["complexity_score"] = text_features.get("complexity_score", 0.5)
            factors["readability_score"] = text_features.get("readability_score", 0.5)
            
            # Pattern-based factors
            if pattern_analysis.get("success"):
                similar_patterns = pattern_analysis.get("similar_patterns", [])
                if similar_patterns:
                    factors["similarity_score"] = np.mean([p["similarity_score"] for p in similar_patterns])
                else:
                    factors["similarity_score"] = 0.0
                
                new_patterns = pattern_analysis.get("new_patterns", [])
                if new_patterns:
                    factors["pattern_quality_score"] = np.mean([p.quality_score for p in new_patterns])
                else:
                    factors["pattern_quality_score"] = 0.5
            else:
                factors["similarity_score"] = 0.0
                factors["pattern_quality_score"] = 0.5
            
            # Structure factors
            factors["structure_score"] = 0.5  # Placeholder - would be calculated from structure analysis
            
            return factors
            
        except Exception as e:
            logger.error(f"Error calculating quality factors: {str(e)}")
            return {"overall_score": 0.5}
    
    def _calculate_predicted_quality(self, quality_factors: Dict[str, float]) -> float:
        """Calculate predicted quality score."""
        try:
            # Weighted combination of factors
            weights = {
                "word_count_score": 0.1,
                "complexity_score": 0.2,
                "readability_score": 0.2,
                "similarity_score": 0.2,
                "pattern_quality_score": 0.2,
                "structure_score": 0.1
            }
            
            predicted_quality = sum(
                quality_factors.get(factor, 0.5) * weight
                for factor, weight in weights.items()
            )
            
            return min(max(predicted_quality, 0.0), 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating predicted quality: {str(e)}")
            return 0.5
    
    def _calculate_prediction_confidence(self, quality_factors: Dict[str, float]) -> float:
        """Calculate prediction confidence."""
        try:
            # Confidence based on factor consistency
            factor_values = list(quality_factors.values())
            if not factor_values:
                return 0.0
            
            # Calculate variance (lower variance = higher confidence)
            variance = np.var(factor_values)
            confidence = max(0.0, 1.0 - variance)
            
            return min(confidence, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating prediction confidence: {str(e)}")
            return 0.5
    
    def _generate_quality_recommendations(self, quality_factors: Dict[str, float]) -> List[str]:
        """Generate quality improvement recommendations."""
        try:
            recommendations = []
            
            if quality_factors.get("readability_score", 0.5) < 0.4:
                recommendations.append("Improve readability by using simpler language and shorter sentences")
            
            if quality_factors.get("complexity_score", 0.5) > 0.8:
                recommendations.append("Consider simplifying complex concepts for better understanding")
            
            if quality_factors.get("similarity_score", 0.5) < 0.3:
                recommendations.append("Add more examples and practical applications similar to successful content")
            
            if quality_factors.get("pattern_quality_score", 0.5) < 0.6:
                recommendations.append("Improve content structure and organization")
            
            if quality_factors.get("word_count_score", 0.5) < 0.3:
                recommendations.append("Expand content with more detailed explanations and examples")
            
            if not recommendations:
                recommendations.append("Content quality looks good - maintain current approach")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
            return ["Unable to generate recommendations"]
    
    def _identify_risk_factors(self, quality_factors: Dict[str, float]) -> List[str]:
        """Identify potential risk factors."""
        try:
            risk_factors = []
            
            if quality_factors.get("readability_score", 0.5) < 0.2:
                risk_factors.append("Very low readability - content may be difficult to understand")
            
            if quality_factors.get("similarity_score", 0.5) < 0.1:
                risk_factors.append("Very low similarity to successful content - may not follow best practices")
            
            if quality_factors.get("pattern_quality_score", 0.5) < 0.3:
                risk_factors.append("Poor content patterns - structure and organization may be inadequate")
            
            if quality_factors.get("word_count_score", 0.5) < 0.2:
                risk_factors.append("Very short content - may lack sufficient detail")
            
            return risk_factors
            
        except Exception as e:
            logger.error(f"Error identifying risk factors: {str(e)}")
            return ["Unable to assess risk factors"]
    
    async def _store_quality_prediction(self, prediction: QualityPrediction):
        """Store quality prediction in the database."""
        try:
            self.quality_collection.add(
                documents=[f"Quality prediction for {prediction.content_id}"],
                metadatas=[{
                    "content_id": prediction.content_id,
                    "predicted_quality": prediction.predicted_quality,
                    "confidence": prediction.confidence,
                    "factors": prediction.factors,
                    "recommendations": prediction.recommendations,
                    "risk_factors": prediction.risk_factors,
                    "timestamp": datetime.utcnow().isoformat()
                }],
                ids=[f"quality_{prediction.content_id}"]
            )
            
        except Exception as e:
            logger.error(f"Error storing quality prediction: {str(e)}")
    
    async def track_performance_metrics(
        self, 
        job_id: str,
        processing_time: float,
        quality_score: float,
        error_count: int,
        human_reviews: int,
        success: bool,
        content_type: str,
        patterns_used: List[str]
    ):
        """Track performance metrics for analysis."""
        try:
            metrics = PerformanceMetrics(
                timestamp=datetime.utcnow(),
                job_id=job_id,
                processing_time=processing_time,
                quality_score=quality_score,
                error_count=error_count,
                human_reviews=human_reviews,
                success=success,
                content_type=content_type,
                patterns_used=patterns_used
            )
            
            # Store metrics
            self.performance_collection.add(
                documents=[f"Performance metrics for {job_id}"],
                metadatas=[{
                    "job_id": job_id,
                    "processing_time": processing_time,
                    "quality_score": quality_score,
                    "error_count": error_count,
                    "human_reviews": human_reviews,
                    "success": success,
                    "content_type": content_type,
                    "patterns_used": patterns_used,
                    "timestamp": metrics.timestamp.isoformat()
                }],
                ids=[f"performance_{job_id}"]
            )
            
            logger.info(f"Performance metrics tracked for job {job_id}")
            
        except Exception as e:
            logger.error(f"Error tracking performance metrics: {str(e)}")
    
    async def get_performance_analytics(
        self, 
        days: int = 30
    ) -> Dict[str, Any]:
        """Get performance analytics for the specified period."""
        try:
            # Calculate date range
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=days)
            
            # Query performance data
            results = self.performance_collection.query(
                query_texts=["performance metrics"],
                n_results=1000,
                where={"timestamp": {"$gte": start_date.isoformat()}}
            )
            
            if not results['metadatas'] or not results['metadatas'][0]:
                return {"error": "No performance data found"}
            
            # Analyze metrics
            metadatas = results['metadatas'][0]
            
            analytics = {
                "period": {
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat(),
                    "days": days
                },
                "summary": {
                    "total_jobs": len(metadatas),
                    "successful_jobs": len([m for m in metadatas if m.get("success", False)]),
                    "failed_jobs": len([m for m in metadatas if not m.get("success", False)]),
                    "average_processing_time": np.mean([m.get("processing_time", 0) for m in metadatas]),
                    "average_quality_score": np.mean([m.get("quality_score", 0) for m in metadatas]),
                    "total_human_reviews": sum([m.get("human_reviews", 0) for m in metadatas])
                },
                "content_type_breakdown": {},
                "quality_trends": {},
                "performance_insights": []
            }
            
            # Content type breakdown
            content_types = {}
            for metadata in metadatas:
                content_type = metadata.get("content_type", "unknown")
                if content_type not in content_types:
                    content_types[content_type] = {"count": 0, "avg_quality": 0, "avg_time": 0}
                content_types[content_type]["count"] += 1
                content_types[content_type]["avg_quality"] += metadata.get("quality_score", 0)
                content_types[content_type]["avg_time"] += metadata.get("processing_time", 0)
            
            for content_type, data in content_types.items():
                if data["count"] > 0:
                    analytics["content_type_breakdown"][content_type] = {
                        "count": data["count"],
                        "average_quality": data["avg_quality"] / data["count"],
                        "average_processing_time": data["avg_time"] / data["count"]
                    }
            
            # Generate insights
            success_rate = analytics["summary"]["successful_jobs"] / analytics["summary"]["total_jobs"] if analytics["summary"]["total_jobs"] > 0 else 0
            if success_rate > 0.9:
                analytics["performance_insights"].append("Excellent success rate - system performing well")
            elif success_rate > 0.7:
                analytics["performance_insights"].append("Good success rate - minor improvements possible")
            else:
                analytics["performance_insights"].append("Success rate needs improvement - investigate failures")
            
            avg_quality = analytics["summary"]["average_quality_score"]
            if avg_quality > 0.8:
                analytics["performance_insights"].append("High quality content generation")
            elif avg_quality > 0.6:
                analytics["performance_insights"].append("Good quality content - room for improvement")
            else:
                analytics["performance_insights"].append("Quality needs improvement - review content generation")
            
            return analytics
            
        except Exception as e:
            logger.error(f"Error getting performance analytics: {str(e)}")
            return {"error": str(e)}
    
    async def get_intelligence_summary(self) -> Dict[str, Any]:
        """Get summary of content intelligence data."""
        try:
            # Get collection counts
            pattern_count = self.pattern_collection.count()
            performance_count = self.performance_collection.count()
            quality_count = self.quality_collection.count()
            
            return {
                "database_status": "active",
                "collections": {
                    "patterns": pattern_count,
                    "performance_metrics": performance_count,
                    "quality_predictions": quality_count
                },
                "embeddings_model": self.embeddings_model,
                "database_path": self.intelligence_db_path,
                "last_updated": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting intelligence summary: {str(e)}")
            return {"error": str(e)}



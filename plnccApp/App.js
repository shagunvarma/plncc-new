import React, { Component } from 'react';
import { StyleSheet, Text, View, TextInput, Button } from 'react-native';
import {createAppContainer} from 'react-navigation'
import {createStackNavigator} from 'react-navigation-stack'
import { Container, Item, Form, Input, Label } from "native-base";
import firebase from 'firebase';

import Loading from './components/Loading'
import SignUp from './components/SignUp'
import Login from './components/LoginScreen'
import Main from './components/Main'
import HomeScreen from './components/HomeScreen'
import CameraPage from './components/testapp/vedo/App'
import ResultsPage from './components/ResultsPage'


const Apps = createStackNavigator(
  
  {
    Loading,
    SignUp,
    Login,
    Main,
    CameraPage,
    HomeScreen,
    ResultsPage,
  },
  {
    initalRouteName: 'Login',
  }
);

const RootStack = createAppContainer(Apps);

export default RootStack;

import React from 'react'
import * as firebase from "firebase";
import { createAppContainer } from 'react-navigation';
import { createStackNavigator } from 'react-navigation-stack';
import { StyleSheet, Text, TextInput, View, Button } from 'react-native'

const firebaseConfig = {
    apiKey: "AIzaSyDdTS13NbVAG9npZdKtd-vxGWHNsuxno18",
    authDomain: "plncc-6c2e9.firebaseapp.com",
    databaseURL: "https://plncc-6c2e9.firebaseio.com",
    projectId: "plncc-6c2e9",
    storageBucket: "plncc-6c2e9.appspot.com",
    messagingSenderId: "1092220463164",
    appId: "1:1092220463164:web:4c3a59a4d1d65d94173b39",
    measurementId: "G-8C81MNZPFB"
};

firebase.initializeApp(firebaseConfig);



export default class Login extends React.Component {
  state = {email: '', password: '', errorMessage: null };



  signUp = (email, pass) => {
      try {
          firebase
              .auth()
              .createUserWithEmailAndPassword(email, pass)
              .then(user => {
                  console.log(user);
              });
          this.props.navigation.navigate('HomeScreen')
      } catch (error) {
          console.log(error.toString())
      }
  };
  handleLogin = (email, pass) => {
      try {
          firebase
              .auth()
              .signInWithEmailAndPassword(email, pass)
              .then(res => {
                  console.log(res.user.email);
              });
          this.props.navigation.navigate('HomeScreen')
      } catch (error) {
          console.log(error.toString())
      }
  };

  render() {

    return (
      <View style = {styles.container}>
        <Text>Login</Text>
        {this.state.errorMessage &&
          <Text style = {{ color: 'red' }}>
            {this.state.errorMessage}
          </Text>}
        <TextInput
          style = {styles.textInput}
          autoCapitalize = "none"
          placeholder = "Email"
          onChangeText = {email => this.setState({email})}
          value = {this.state.email}
        />
        <TextInput
          secureTextEntry
          style = {styles.textInput}
          autoCapitalize = "none"
          placeholder = "Password"
          onChangeText = {password => this.setState({password})}
          value = {this.state.password}
        />
        <Button title="Login" onPress={() => this.handleLogin(this.state.email, this.state.password)} />
        <Button
          title="Sign Up"
          onPress={() => this.signUp(this.state.email, this.state.password)}
        />
        </View>
    )
  }
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center'
  },
  textInput: {
    height: 40,
    width: '90%',
    borderColor: 'gray',
    borderWidth: 1,
    marginTop: 8
  }
})
